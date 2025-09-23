
1. Clone the repository
git clone https://github.com/BenJMurarotto/deep-learning-alzheimers.git
cd deep-learning-alzheimers/frontend

2. Create virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate

3. Install dependancies
pip install -r requirements_inference.txt

4. Download the pretrained model and unzip

5. Run the app
streamlit run app_streamlit.py