"""Test a built wheel against its sdist in a fresh non-editable environment."""
from __future__ import annotations
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import venv
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    parser.add_argument('sdist', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    wheel, sdist = args.wheel.resolve(), args.sdist.resolve()
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    with tempfile.TemporaryDirectory(prefix='asm-installed-') as directory:
        root = Path(directory)
        source = root/'source'; source.mkdir()
        with tarfile.open(sdist) as archive:
            if len(archive.getmembers()) > 10000 or sum(m.size for m in archive.getmembers()) > 100*1024*1024:
                raise ValueError('unexpected distribution size')
            archive.extractall(source, filter='data')
        projects = list(source.iterdir())
        if len(projects) != 1 or not projects[0].is_dir():
            raise ValueError('expected exactly one source distribution directory')
        project = projects[0]
        metadata = tomllib.loads((project/'pyproject.toml').read_text())['project']
        package = project/'src/alignment_metaplasticity'
        reference = {'alignment_metaplasticity/'+p.relative_to(package).as_posix(): p.read_bytes()
                     for p in package.rglob('*.py')}
        with zipfile.ZipFile(wheel) as archive:
            actual = {n: archive.read(n) for n in archive.namelist() if n.startswith('alignment_metaplasticity/') and n.endswith('.py')}
            candidates = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            if len(candidates) != 1: raise ValueError('ambiguous wheel metadata')
            identity = BytesParser().parsebytes(archive.read(candidates[0]))
            if identity.get_all('Name') != [metadata['name']] or identity.get_all('Version') != [metadata['version']]:
                raise ValueError('wheel identity differs from source metadata')
            if actual != reference: raise ValueError('wheel runtime differs from source distribution')
        environment = root/'venv'; venv.EnvBuilder(with_pip=True).create(environment)
        python = environment/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        def run(arguments):
            subprocess.run([str(python), *arguments], cwd=project, env=env, check=True, timeout=300)
        if sys.platform == 'linux':
            run(['-m', 'pip', 'install', 'torch>=2.6,<3', '--index-url', 'https://download.pytorch.org/whl/cpu'])
        run(['-m', 'pip', 'install', str(wheel), 'pytest>=8,<10'])
        run(['-m', 'pip', 'check'])
        run(['-c', 'import alignment_metaplasticity as a,pathlib,sys,importlib.metadata as m; p=pathlib.Path(a.__file__).resolve(); assert pathlib.Path(sys.prefix).resolve() in p.parents; assert m.version(sys.argv[1])==sys.argv[2]; print("Installed runtime",p)', metadata['name'], metadata['version']])
        run(['-m', 'pytest', '-q'])
        run(['scripts/verify_calibration.py', 'results/calibrated-transfer-v2'])
        run(['-m', 'experiments.portfolio_demo', '--out', str(root/'demo')])
        report = {'status': 'PASS', 'distribution': metadata['name'], 'version': metadata['version'],
                  'runtime_modules_matched': len(reference), 'installation': 'fresh non-editable wheel outside checkout',
                  'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
                  'sdist_sha256': hashlib.sha256(sdist.read_bytes()).hexdigest()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
