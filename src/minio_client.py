# Example for using MinIO provided Dimitrios Fotiou from AUTH 
from minio import Minio
from minio.error import S3Error
from PIL import Image
import numpy as np
import io
import os
import logging

MINIO_ENDPOINT= 'storage.tema.digital-enabler.eng.it:443'
MINIO_ACCESS_KEY = "AUMFK4CGDFORW7PC9URA"     # Used in all examples
MINIO_SECRET= "v9L6zs+G8Qu0UKgfMi8FNIncXtZ+ASMJrAXQwpTB"      # Used in all examples
FHHI_MINIO_BUCKET='fhhi'
NAPLES_MINIO_BUCKET='naples'
logger = logging.getLogger(__name__)

class MinIOClient:
    def __init__(self, url=MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET, secure=True):
        self.client = Minio(
            url,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure #if HTTPS is used, set this to True
        )

    def upload_file(self, bucket_name, object_name, file_path):
        logger.info("MinIO upload_file start bucket=%s object=%s file=%s", bucket_name, object_name, file_path)
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)
            print("Bucket created successfully.")
            logger.info("MinIO bucket created bucket=%s", bucket_name)
        try:
            # Remove leading slash from object_name
            if object_name.startswith("/"):
                object_name = object_name[1:]
            file_size = os.path.getsize(file_path)
            logger.info(
                "MinIO upload_file sending bucket=%s object=%s bytes=%s",
                bucket_name,
                object_name,
                file_size,
            )

            results = self.client.fput_object(
                bucket_name,
                object_name,
                file_path
            )
            print(results.object_name)
            print(results.etag)
            print("File uploaded successfully.")
            logger.info(
                "MinIO upload_file success bucket=%s object=%s etag=%s",
                bucket_name,
                results.object_name,
                results.etag,
            )
        except S3Error as exc:
            print("Error occurred.", exc)
            logger.exception("MinIO upload_file failed bucket=%s object=%s file=%s", bucket_name, object_name, file_path)
            raise

    def upload_text(self, bucket_name, object_name, text: str):
        logger.info("MinIO upload_text start bucket=%s object=%s", bucket_name, object_name)
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print("Bucket created successfully.")
                logger.info("MinIO bucket created bucket=%s", bucket_name)

            if object_name.startswith("/"):
                object_name = object_name[1:]

            payload = text.encode("utf-8")
            logger.info(
                "MinIO upload_text sending bucket=%s object=%s bytes=%s",
                bucket_name,
                object_name,
                len(payload),
            )
            buffer = io.BytesIO(payload)
            results = self.client.put_object(
                bucket_name,
                object_name,
                buffer,
                len(payload),
                content_type="text/plain; charset=utf-8",
            )
            print(f"Text '{results.object_name}' uploaded successfully.")
            print(f"Etag: {results.etag}")
            logger.info(
                "MinIO upload_text success bucket=%s object=%s etag=%s",
                bucket_name,
                results.object_name,
                results.etag,
            )
            return results
        except S3Error as exc:
            print("Error occurred.", exc)
            logger.exception("MinIO upload_text failed bucket=%s object=%s", bucket_name, object_name)
            raise

    def upload_text_file(self, bucket_name, object_name, file_path):
        logger.info("MinIO upload_text_file start bucket=%s object=%s file=%s", bucket_name, object_name, file_path)
        if not os.path.exists(file_path):
            logger.error("MinIO upload_text_file missing file=%s", file_path)
            raise FileNotFoundError(file_path)
        file_size = os.path.getsize(file_path)
        logger.info(
            "MinIO upload_text_file reading file=%s bytes=%s",
            file_path,
            file_size,
        )
        with open(file_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        logger.info(
            "MinIO upload_text_file loaded file=%s chars=%s",
            file_path,
            len(text),
        )
        return self.upload_text(bucket_name, object_name, text)

    def download_image(self, bucket_name, object_name):
        try:
            data = self.client.get_object(bucket_name, object_name)
            
            # Create a bytes buffer
            buffer = io.BytesIO()

            # Write the streamed data to the buffer
            for d in data.stream(32*1024):
                buffer.write(d)

            # Go to the start of the buffer
            buffer.seek(0)

            # Open the image from the buffer
            img = Image.open(buffer)

            # Convert the image to a numpy array
            img_array = np.array(img)

            print("Image downloaded and converted to numpy array successfully.")
            return img_array
        except S3Error as exc:
            print("Error occurred.", exc)
            
        
    def upload_image(self, bucket_name, object_name, image: np.ndarray):
        """
        Upload a numpy array image to MinIO.
    
        Args:
            bucket_name (str): Name of the bucket
            object_name (str): Name of the object to create in MinIO
            image (np.ndarray): Numpy array representation of the image
        """
        try:
            # Create bucket if it doesn't exist
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                print("Bucket created successfully.")
            
            # Remove leading slash from object_name
            if object_name.startswith("/"):
                object_name = object_name[1:]
            
            # Convert numpy array to PIL Image
            img = Image.fromarray(image)
            
            # Create a bytes buffer
            buffer = io.BytesIO()
            
            # Save image to the buffer in PNG format
            img.save(buffer, format='PNG')
            
            # Reset buffer position to start
            buffer.seek(0)
            
            # Get buffer size
            size = buffer.getbuffer().nbytes
            
            # Upload the image buffer to MinIO
            results = self.client.put_object(
                bucket_name,
                object_name,
                buffer,
                size,
                content_type='image/png'
            )
            
            print(f"Image '{results.object_name}' uploaded successfully.")
            print(f"Etag: {results.etag}")
            
        except S3Error as exc:
            print("Error occurred.", exc)
        except Exception as ex:
            print(f"Unexpected error occurred: {ex}") 
