# Piyus563-AI-Document-Analyzer

## Free Hosting On PythonAnywhere

Use PythonAnywhere if Render asks for payment.

1. Create a free account at https://www.pythonanywhere.com
2. Open a Bash console and run:

```bash
git clone https://github.com/Piyus563/Piyus563-AI-Document-Analyzer.git
cd Piyus563-AI-Document-Analyzer
pip3 install --user -r backend/requirements.txt
```

3. Go to Web > Add a new web app.
4. Choose Manual configuration.
5. Choose Python 3.10 or newer.
6. Open the WSGI configuration file.
7. Paste this code, replacing `YOUR_USERNAME` with your PythonAnywhere username:

```python
import sys
from pathlib import Path

USERNAME = "YOUR_USERNAME"
PROJECT_DIR = Path(f"/home/{USERNAME}/Piyus563-AI-Document-Analyzer")

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app import app as application
```

8. Click Reload.
