# Submission Ration Prediction App
## Steps to run the app:
1. install requirements.txt
2. run server.py if running in pycharm
      or
   python -m api.server.py
3. Flask app will run on http://localhost:5432
4. For training just ensure that a csv file named 'data.csv' is placed in data folder
   API: GET : http://localhost:5432/train, I kept an empty file, please replace with the file provided to me. Github not allowing to load big files.
5. For inference pass data as a json in Post command for API
   API: POST : http://localhost:5432/inference
   Data Format should match CSV format supplied for assignment

