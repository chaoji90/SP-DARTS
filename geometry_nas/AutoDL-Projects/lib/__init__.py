import sys
from pathlib import Path
from pathlib import Path
lib_dir = (Path(__file__).parent).resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))

