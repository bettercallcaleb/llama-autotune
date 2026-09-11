from dataclasses import asdict,dataclass
from pathlib import Path
import platform
from ..config import AutotuneError,digest
from ..executor import Executor
from ..storage.store import Store
from ..llama.capabilities import discover,find_binary

@dataclass
class BuildPlan:
    source: str
    directory: str
    backend: str
    definitions: list[str]
    diagnostics: list[str]

    def json(self) -> dict:
        return asdict(self)

def plan(env: dict,source: Path,directory: Path,backend: str) -> BuildPlan:
    source,directory=source.expanduser().resolve(),directory.expanduser().resolve()
    if not (source/"CMakeLists.txt").is_file() or not (source/"ggml").is_dir():
        raise AutotuneError(f"Not a llama.cpp source tree: {source}; clone ggml-org/llama.cpp first")
    if source==directory:
        raise AutotuneError("Use a separate build directory")
    diagnostics=[]
    if backend=="auto":
        backend="cuda" if env["gpus"] and env["tools"]["nvcc"]["ok"] else "cpu"
        if env["gpus"] and backend=="cpu":
            diagnostics.append("NVIDIA detected but nvcc unavailable; selected CPU. Install a compatible CUDA toolkit to enable CUDA.")
    missing=[name for name in ("cmake",) if not env["tools"][name]["ok"]]
    if not any(env["tools"][x]["ok"] for x in ("gcc","clang")):
        missing.append("C/C++ compiler (gcc/g++ or clang/clang++)")
    if backend=="cuda" and not env["tools"]["nvcc"]["ok"]:
        missing.append("CUDA toolkit including nvcc")
    if missing:
        raise AutotuneError("Install required dependencies using your OS package manager: "+", ".join(missing))
    definitions=["-DCMAKE_BUILD_TYPE=Release","-DGGML_NATIVE=ON","-DLLAMA_BUILD_SERVER=ON",
                 "-DLLAMA_BUILD_EXAMPLES=ON","-DLLAMA_BUILD_TESTS=OFF","-DGGML_CUDA="+("ON" if backend=="cuda" else "OFF"),
                 "-DGGML_HIP=OFF","-DGGML_VULKAN=OFF","-DGGML_SYCL=OFF","-DGGML_METAL=OFF"]
    if backend=="cuda":
        archs=sorted({g["compute_capability"].replace(".","") for g in env["gpus"] if g.get("compute_capability") and g["compute_capability"].replace(".","").isdigit()})
        if archs:
            definitions.append("-DCMAKE_CUDA_ARCHITECTURES="+";".join(archs))
    if env["tools"]["ccache"]["ok"]:
        definitions += ["-DCMAKE_C_COMPILER_LAUNCHER=ccache","-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    return BuildPlan(str(source),str(directory),backend,definitions,diagnostics)

def compile_plan(p: BuildPlan,env: dict,ex: Executor,store: Store,jobs: int,timeout: float) -> dict:
    if platform.system()!="Linux":
        raise AutotuneError("v0 compilation supports Linux only")
    source,directory=Path(p.source),Path(p.directory)
    commit=ex.run(["git","-C",source,"rev-parse","HEAD"]).stdout.strip()
    status=ex.run(["git","-C",source,"status","--porcelain","--untracked-files=all"]).stdout
    diff=ex.run(["git","-C",source,"diff","HEAD","--binary"])
    manifest={"plan":p.json(),"environment":env,"commit":commit or None,"dirty":status,
              "diff_hash":digest(diff.stdout),"diff_log":diff.log,"jobs":jobs}
    ex.run(["cmake","-S",source,"-B",directory,*p.definitions],timeout=timeout,check=True)
    ex.run(["cmake","--build",directory,"--config","Release","--parallel",str(jobs),"--target","llama-server","llama-bench"],timeout=timeout,check=True)
    manifest["cmake_cache"]=(directory/"CMakeCache.txt").read_text(errors="replace")
    binary=find_binary("llama-server",None,str(directory))
    if binary is None:
        raise AutotuneError("Build succeeded but llama-server binary could not be located")
    manifest["server_fingerprint"]=discover(binary,ex).fingerprint
    manifest["fingerprint"]=digest(manifest)
    store.write("builds",manifest["fingerprint"],manifest)
    store.write("builds","latest",manifest)
    return manifest
