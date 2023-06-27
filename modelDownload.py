import os
import requests
from tqdm import tqdm

def downloadModel(url: str, dest_folder: str):
    if not os.path.exists(dest_folder):
        os.makedirs("./models/", exist_ok=True) 

    filename = url.split("/")[-1].replace(" ", "_")
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("Saving to", os.path.abspath(file_path))
        total_size_in_bytes= int(r.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(block_size):
                progress_bar.update(len(chunk))
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
    else:
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))