import cProfile
import pstats

def extract_pstats_cumtime_ranking(cProfile_output_name, cProfile_time_name):
    with open(cProfile_time_name, "w") as f:
        p = pstats.Stats(cProfile_output_name, stream=f)
        p.sort_stats("cumtime").print_stats()

def extract_pstats_calls_ranking(cProfile_output_name, cProfile_calls_name):
    with open(cProfile_calls_name, "w") as f:
        p = pstats.Stats(cProfile_output_name, stream=f)
        p.sort_stats("calls").print_stats()


def cProfile_run_and_analyze(run_func_str):
    cProfile_output_name, cProfile_time_name, cProfile_calls_name, cProfile_percentage_png_name = "tmp_CProfile_output_name.pstats", "tmp_cProfile_time_name.txt", "tmp_cProfile_calls_name.txt", "tmp_cProfile_time_percentage_name.png"

    cProfile.run(run_func_str, cProfile_output_name)

    # extract cumtime statistics and print into a txt file
    extract_pstats_cumtime_ranking(cProfile_output_name, cProfile_time_name)

    # extract calls statistics and print into a txt file
    extract_pstats_calls_ranking(cProfile_output_name, cProfile_calls_name)

    # generate a png file for cumtime statistics
    # !gprof2dot -f pstats tmp_CProfile_output_name.pstats | dot -Tpng -o tmp_pstats_output.png

import psutil
import math
import requests

bytes_per_GB = 1024**3

def convert_bytes_to_GB(x_bytes):
    x_GB = x_bytes / bytes_per_GB
    return x_GB

def convert_GB_to_bytes(x_GB):
    x_bytes = x_GB * bytes_per_GB
    return x_bytes

def get_RAM_available_in_bytes():
    return psutil.virtual_memory()[1]

def get_RAM_available_in_GB():
    return convert_bytes_to_GB(get_RAM_available_in_bytes())

def get_RAM_used_in_bytes():
    return psutil.virtual_memory()[3]

def get_RAM_used_in_GB():
    return convert_bytes_to_GB(get_RAM_used_in_bytes())

def round_down_n_decimal_places(a, n):
    return math.floor(a * 10**n) / 10**n

def download_file_from_google_drive(id, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id , 'confirm': 1 }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    # link: https://stackoverflow.com/a/39225272/5040208
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    # link: https://stackoverflow.com/a/39225272/5040208
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)