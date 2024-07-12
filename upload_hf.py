from huggingface_hub import HfApi, HfFolder
import numpy as np
import os



# Your Hugging Face username and repository name
repo_id = "wufuheng/VQ"  # Replace with your information

def upload(path_or_fileobj="256_ffhq.npz", path_in_repo="256_ffhq.npz"):
    # Authenticate using your token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        HfFolder.save_token(hf_token)
    else:
        raise ValueError("HF_TOKEN environment variable is not set.")

    # Initialize the API
    api = HfApi()
    # Create repository (optional, only if you haven't created it yet)
    #create_repo(repo_id, repo_type="dataset", private=True)

    # Upload file
    api.upload_file(
        path_or_fileobj=path_or_fileobj,
        path_in_repo=path_in_repo,  # Keep the filename consistent
        repo_id=repo_id,
    )





def get_features(repo_id = "wufuheng/fid_features", filename = "512_ffhq.npz"):
    from huggingface_hub import hf_hub_download

    try:
        # Download the file from Hugging Face repository to cache
        file_path = hf_hub_download(repo_id, filename, repo_type="dataset")
        # Load the file using NumPy
        data = np.load(file_path)
        if 'features' in data:
            features = data['features']
        else:
            print("Error: 'features' key not found in the loaded file. Please check the file content.")
            return None
    except Exception as err:
        print(f"Error occurred: {err}")
        return None

    return features


if __name__ == "__main__":
    #f = get_features()
    #print(f.shape)
    upload(path_or_fileobj="mbin/sit-lc-100K-f16.pth", path_in_repo="mbin/sit-lc-100K-f16.pth")