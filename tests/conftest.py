import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
warnings.filterwarnings("ignore")
