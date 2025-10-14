import os
import cv2
from typing import Any
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding




def save_image_locally(folder_path: str, image_path: str, image: Any) -> None:
    """
    Saves an image to the specified local folder path.

    Args:
        folder_path (str): The directory path where the image will be saved.
        image_path (str): The name of the image file (including extension) to be saved.
        image (Any): The image to be saved. This can be a NumPy array representing the image.

    This function checks if the specified folder exists and creates it if not. It then saves the provided 
    image to the specified path within that folder using OpenCV's `imwrite` function.
    """
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create the folder if it does not exist
    cv2.imwrite(os.path.join(folder_path, image_path), image)
    

def delete_files(files_list):
    for file in files_list:
        if os.path.exists(file):
            os.remove(file)
   
def decrypt_models(key, input_file,output_file=None):
    try:
        salt = key.encode('utf-8')
        with open(input_file, 'rb') as f:
            data = f.read()
        # Extract the IV from the first 16 bytes
        
        iv = data[:16]
        ciphertext = data[16:]
        # Create cipher
        cipher = Cipher(algorithms.AES(salt), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()

        # Decrypt the ciphertext
        decrypted_data = decryptor.update(ciphertext) + decryptor.finalize()

        # Unpad the decrypted data
        unpadder = padding.PKCS7(128).unpadder()
        decrypted_data = unpadder.update(decrypted_data) + unpadder.finalize()

        if output_file is not None:
            with open(output_file, 'wb') as f:
                f.write(decrypted_data)
            return output_file
        
        return decrypted_data
    except Exception as e:
        print("Error while decryting model: ", e)
        return None         
            


