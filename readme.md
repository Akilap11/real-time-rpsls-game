
# Setting Up and Running a Flask Application

Follow these steps to set up and run a Flask application in your project:

## 1. Create a Virtual Environment
Run the following command to create a virtual environment:
```bash
python -m venv venv
```

## 2. Set Execution Policy (Windows Only)
If you are on Windows, set the execution policy to allow scripts to run:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

## 3. Activate the Virtual Environment
Activate the virtual environment using the following command:
```bash
.\venv\Scripts\activate
```

## 4. Upgrade `pip`
Ensure you have the latest version of `pip` installed:
```bash
python -m pip install --upgrade pip
```

## 5. Install Flask
Install Flask, the web framework:
```bash
python -m pip install flask
```

## 6. Run the Application
Run your Flask application:
```bash
python app.py
```

Your Flask application should now be running. Open your browser and navigate to the URL provided in the terminal (usually `http://127.0.0.1:5000`).
```
