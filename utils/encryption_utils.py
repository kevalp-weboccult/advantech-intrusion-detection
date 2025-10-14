"""
    This file contains function for encrypting or decrypting files 
"""
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    # from logger import logger 
except:
    raise ImportError("Error importing dependencies of encrption file")


def encrypt_file(key, input_file,file_data,output_file):
    try:
        
        cipher = Fernet(key)
        if input_file:
            with open(input_file, 'rb') as file:
                file_data = file.read()
        # Convert string data to bytes if needed
        if isinstance(file_data, str):
            file_data = file_data.encode()
        ciphertext = cipher.encrypt(file_data)
        # print(ciphertext)
        with open(output_file, 'wb') as file:
            file.write(ciphertext)
        return True
    except Exception as e:
        print("Error in encrypting file",e)
        return False
        

def decrypt_file(key, input_file,output_file=None):
    try:
        cipher = Fernet(key)

        with open(input_file, 'rb') as file:
            ciphertext = file.read()

        plaintext = cipher.decrypt(ciphertext)
        if output_file is not None:
            with open(output_file, 'wb') as file:
                file.write(plaintext)
            return output_file

        return plaintext
    except Exception as e:
        print("Error while decryption: ", e)
        return None
    
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

    