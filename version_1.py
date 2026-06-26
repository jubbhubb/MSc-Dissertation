from preprocessing import create_experiment_folder
from api_calls import process_folder
from datetime import datetime


def main():
    create_experiment_folder("testing_006", lowercase=True)
    process_folder("experiments/testing_006")
    
main()