# ClientSense_backend

This is the backend api of the ClientSense sales assisting webapp powered by Cohere done using fastapi.

It makes use of Cohere AI APIs for : summarize, toxicity detection, sentiment analysis and suggession for sales.

You can clone this git repo, and get it running by the following steps:
1.) git clone the repo
2.) create a virtual environment : python3 -m venv env
3.) activate the virtual environment: source env/bin/activate
4.) install the libraries mentioned in the requirements file : pip install requirements.txt
5.) Run the server : uvicorn main:app --reload