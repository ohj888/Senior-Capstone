"""Download the original ConceptGraphs runtime for an isolated local WSL install.

Run with the Windows Python that has network access. Downloads are stored in
dependencies/conceptgraphs-runtime; Linux installation uses only local files.
Existing Windows model-training packages and adapter weights are untouched.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import time
import urllib.request
import urllib.error
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "dependencies/conceptgraphs-runtime"
DISTRO = "Ubuntu"


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _download_once(url, target, md5=None, sha256=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = target.with_name(target.name + ".download.json")
    if target.is_file():
        if sha256 and digest(target) == sha256 or md5 and digest(target, "md5") == md5:
            return target
        if not sha256 and not md5 and meta.is_file():
            saved = json.loads(meta.read_text())
            if saved["url"] == url and saved["bytes"] == target.stat().st_size and saved["sha256"] == digest(target):
                return target
        raise ValueError(f"Existing download has no matching integrity record: {target}")
    partial = target.with_name(target.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"User-Agent": "ConceptGraphs-local-setup", **({"Range": f"bytes={offset}-"} if offset else {})})
    print("Downloading", target.name, flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise IOError(f"Server returned the wrong resume range for {target.name}")
        expected = response.headers.get("Content-Length")
        size = 0
        with partial.open("ab" if append else "wb") as handle:
            while chunk := response.read1(64*1024):
                handle.write(chunk)
                size += len(chunk)
        if expected is not None and size != int(expected):
            raise IOError(f"Incomplete download: {target.name}")
    if md5 and digest(partial, "md5") != md5:
        raise IOError(f"MD5 mismatch: {target.name}")
    checksum = digest(partial)
    if sha256 and checksum != sha256:
        raise IOError(f"SHA256 mismatch: {target.name}")
    partial.replace(target)
    meta.write_text(json.dumps({"url": url, "bytes": target.stat().st_size, "sha256": checksum}, indent=2), encoding="utf-8")
    print("Ready", target.name, target.stat().st_size, "bytes", flush=True)
    return target


def download(url, target, md5=None, sha256=None):
    for attempt in range(3):
        try:
            return _download_once(url, target, md5, sha256)
        except (TimeoutError, ConnectionError, urllib.error.URLError) as exc:
            if attempt == 2:
                raise
            print("Retrying", Path(target).name, type(exc).__name__, flush=True)
            time.sleep(2)


def conda_packages():
    plan = json.loads((RUNTIME / "conda-plan-cuda118.json").read_text(encoding="utf-8-sig"))
    if not plan.get("success"):
        raise ValueError("Conda solve failed; inspect conda-plan.json")
    # LINK includes packages already in conda's cache; FETCH alone can omit
    # them on a repeated solve and produce an incomplete offline environment.
    fetched = {p["name"]: p for p in plan["actions"].get("FETCH", [])}
    packages = []
    for linked in plan["actions"].get("LINK", []):
        record = fetched.get(linked["name"])
        if record is None:
            cached = RUNTIME / "conda-cache" / linked["dist_name"] / "info/repodata_record.json"
            if not cached.is_file():
                raise RuntimeError(f"Missing archive metadata for {linked['dist_name']}; repeat resolve with a fresh runtime cache")
            record = json.loads(cached.read_text())
        packages.append(record)
    def fetch(package):
        filename = package["fn"]
        return download(package["url"],
                        RUNTIME / "conda-packages" / filename, package.get("md5"), package.get("sha256"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(fetch, packages))
    # Explicit local archives avoid WSL's broken internet route and re-solving.
    linux = linux_path(RUNTIME / "conda-packages")
    (RUNTIME / "conda-explicit.txt").write_text("@EXPLICIT\n" + "".join(f"file://{quote(linux + '/' + path.name)}\n" for path in files), encoding="utf-8")


def resolve():
    candidates = [p / "Scripts/conda.exe" for p in (Path(sys.prefix), *Path(sys.prefix).parents)]
    executable = next((p for p in candidates if p.is_file()), None)
    if executable is None:
        raise FileNotFoundError("Run setup with a Windows Anaconda Python so conda.exe can be located.")
    env = {**os.environ, "CONDA_PKGS_DIRS": str(RUNTIME / "conda-cache")}
    command = [str(executable), "create", "--dry-run", "--json", "--platform", "linux-64", "--override-channels",
               "-c", "conda-forge", "-c", "nvidia", "-p", str(RUNTIME / "solve-only"), "python=3.10", "pip=24.0",
               "numpy=1.26.4", "gxx_linux-64=11", "cmake", "ninja", "cuda-nvcc=11.8", "cuda-cudart-dev=11.8",
               "cuda-cudart=11.8", "cuda-cccl=11.8"]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    (RUNTIME / "conda-plan-cuda118.json").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


def pip_packages():
    wheels = RUNTIME / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    # These are pure-Python source distributions, so their wheels can be built
    # on Windows and then installed offline by Linux Python.
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheels),
                    "antlr4-python3-runtime==4.9.3", "fvcore==0.1.5.post20221221", "iopath==0.1.10", "fairscale==0.4.13",
                    "pathtools==0.1.2", "lit==17.0.1"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "download", "--dest", str(wheels), "--find-links", str(wheels),
                    "--platform", "manylinux2014_x86_64", "--platform", "manylinux_2_28_x86_64",
                    "--platform", "linux_x86_64", "--python-version", "310", "--implementation", "cp",
                    "--abi", "cp310", "--abi", "abi3", "--only-binary=:all:",
                    "-r", str(Path(__file__).with_name("conceptgraphs-runtime-requirements.txt"))], check=True)


def source(repo, ref, name):
    target = RUNTIME / "src" / name
    if (target / ".source.json").is_file():
        return target
    if len(ref) != 40:
        resolved = subprocess.check_output(["git", "ls-remote", f"https://github.com/{repo}.git", f"refs/heads/{ref}"], text=True).strip()
        if not resolved:
            raise ValueError(f"Missing source ref: {repo}/{ref}")
        ref = resolved.split()[0]
    archive = download(f"https://codeload.github.com/{repo}/tar.gz/{ref}", RUNTIME / "archives" / f"{name}-{ref}.tar.gz")
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = Path(member.name).parts[1:]
            if not parts or member.isdir():
                continue
            if not member.isfile():
                continue  # No archive symlinks or devices are created.
            destination = root.joinpath(*parts).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("Archive path escapes source directory")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as incoming, destination.open("wb") as output:
                for chunk in iter(lambda: incoming.read(1024*1024), b""):
                    output.write(chunk)
    (target / ".source.json").write_text(json.dumps({"repo": repo, "commit": ref}), encoding="utf-8")
    print("Source ready:", name, ref, flush=True)
    return target


def sources():
    items = [("IDEA-Research/Grounded-Segment-Anything", "a4d76a2b55e348943cba4cd57d7553c354296223", "Grounded-Segment-Anything"),
             ("gradslam/gradslam", "59ca872e3d265ad09f63c4793d011fad67064452", "gradslam"),
             ("krrish94/chamferdist", "ee753899b583bdc06bd2505dc162fd54ba0c6c5b", "chamferdist"),
             ("xinyu1205/recognize-anything", "7cb804a8609e9f4b1a50b7f31436d2df40bb9481", "recognize-anything")]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: source(*item), items))


def torch_wheels():
    items = [
        ("https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp310-cp310-linux_x86_64.whl", "torch-2.0.1+cu118-cp310-cp310-linux_x86_64.whl"),
        ("https://download.pytorch.org/whl/cu118/torchvision-0.15.2%2Bcu118-cp310-cp310-linux_x86_64.whl", "torchvision-0.15.2+cu118-cp310-cp310-linux_x86_64.whl"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda item: download(item[0], RUNTIME / "wheels" / item[1]), items))


def weights():
    items = [
        ("https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth", "groundingdino_swint_ogc.pth"),
        ("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", "sam_vit_h_4b8939.pth"),
        ("https://huggingface.co/spaces/xinyu1205/Recognize_Anything-Tag2Text/resolve/main/ram_swin_large_14m.pth", "ram_swin_large_14m.pth"),
        ("https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin", "clip_vit_h14.bin"),
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda item: download(item[0], RUNTIME / "weights" / item[1]), items))
    # Both GroundingDINO and RAM need BERT files even though the scene graph
    # semantics later come from Qwen/SmolVLM.
    for filename in ("config.json", "vocab.txt", "tokenizer_config.json", "tokenizer.json", "pytorch_model.bin"):
        download("https://huggingface.co/google-bert/bert-base-uncased/resolve/"
                 "86b5e0934494bd15c9632b12f734a8a67f723594/" + filename,
                 RUNTIME / "weights/bert-base-uncased" / filename)


def linux_path(path):
    return subprocess.check_output(["wsl.exe", "-d", DISTRO, "--", "wslpath", "-u", str(Path(path).resolve())], text=True).strip()


def install(args):
    """Install only into the named isolated WSL environment, from local files."""
    prefix, conda = args.env_prefix, args.linux_conda
    launch = ["wsl.exe", "-d", DISTRO, "--"]
    # Explicit conda archives are self-contained and do not invoke a solver.
    exists = subprocess.run([*launch, "test", "-x", prefix + "/bin/python"]).returncode == 0
    subprocess.run([*launch, conda, "install" if exists else "create", "--yes", "--offline", "--prefix", prefix,
                    "--file", linux_path(RUNTIME / "conda-explicit.txt")], check=True)
    pip = [*launch, conda, "run", "--no-capture-output", "-p", prefix, "env",
           f"CUDA_HOME={prefix}", f"MAX_JOBS={args.build_jobs}", f"TORCH_CUDA_ARCH_LIST={args.cuda_arch}",
           "FORCE_CUDA=1", "PIP_NO_INDEX=1", f"PIP_FIND_LINKS={linux_path(RUNTIME / 'wheels')}",
           prefix + "/bin/python", "-m", "pip", "install", "--no-index",
           "--find-links", linux_path(RUNTIME / "wheels")]
    subprocess.run([*pip, "-r", linux_path(Path(__file__).with_name("conceptgraphs-runtime-requirements.txt"))], check=True)
    # PyTorch3D's published binary matches Python 3.10 / Torch 2.0.1 / CUDA 11.8.
    package = RUNTIME / "conda-packages/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2"
    subprocess.run([*launch, conda, "install", "--yes", "--offline", "--no-deps", "-p", prefix, linux_path(package)], check=True)
    for relative in ("src/chamferdist", "src/gradslam", "src/Grounded-Segment-Anything/GroundingDINO",
                     "src/Grounded-Segment-Anything/segment_anything", "src/recognize-anything"):
        subprocess.run([*pip, "--no-build-isolation", "--no-deps", linux_path(RUNTIME / relative)], check=True)
    subprocess.run([*pip, "--no-build-isolation", "--no-deps", "-e", linux_path(ROOT / "dependencies/concept-graphs")], check=True)
    check = "import torch,cv2,open3d,open_clip,faiss,pytorch3d,groundingdino,segment_anything,ram,gradslam; print('Mapping imports OK; CUDA:',torch.cuda.is_available())"
    subprocess.run([*launch, prefix + "/bin/python", "-c", check], check=True)
    (RUNTIME / "installed.json").write_text(json.dumps({"distro": DISTRO, "python": prefix + "/bin/python",
                                                       "cuda_arch": args.cuda_arch}, indent=2), encoding="utf-8")


def pytorch3d():
    filename = "pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2"
    download("https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/" + filename,
             RUNTIME / "conda-packages" / filename)


def main():
    global DISTRO, RUNTIME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["all", "resolve", "conda", "sources", "torch", "pip", "weights", "pytorch3d", "install"])
    parser.add_argument("--distro", default="Ubuntu")
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME)
    parser.add_argument("--env-prefix", help="Isolated WSL environment; defaults to ~/miniconda3/envs/conceptgraphs-pilot")
    parser.add_argument("--linux-conda", help="WSL conda executable; defaults to ~/miniconda3/bin/conda")
    parser.add_argument("--build-jobs", type=int, default=2)
    parser.add_argument("--cuda-arch", default="8.9", help="RTX 4060 default; change for another local GPU")
    args = parser.parse_args()
    DISTRO, RUNTIME = args.distro, args.runtime_root.resolve()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if args.action in ("install", "all"):
        home = subprocess.check_output(["wsl.exe", "-d", DISTRO, "--", "printenv", "HOME"], text=True).strip()
        args.env_prefix = args.env_prefix or home + "/miniconda3/envs/conceptgraphs-pilot"
        args.linux_conda = args.linux_conda or home + "/miniconda3/bin/conda"
        if args.env_prefix.rstrip("/") in (home, "", str(PurePosixPath(args.linux_conda).parent.parent)):
            raise ValueError("Choose an isolated environment, not the home or base environment")
    actions = {"resolve": resolve, "conda": conda_packages, "sources": sources, "torch": torch_wheels,
               "pip": pip_packages, "weights": weights, "pytorch3d": pytorch3d, "install": lambda: install(args)}
    for name in (actions if args.action == "all" else (args.action,)):
        if name == "resolve" and args.action == "all":
            saved_plan = RUNTIME / "conda-plan-cuda118.json"
            if saved_plan.is_file() and json.loads(saved_plan.read_text(encoding="utf-8-sig")).get("success"):
                print("Reusing saved conda package plan; use action=resolve to refresh it", flush=True)
                continue
        print("Setup stage:", name, flush=True)
        actions[name]()


if __name__ == "__main__":
    main()
