import time
import pika
import json
import traceback
from threading import Thread
from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
# from constant import HIDE_SETTINGS,DEBUG_MODE,RABBITMQ_USERNAME,RABBITMQ_PASSWORD,RABBITMQ_HOST,RABBITMQ_PORT,RABBITMQ_QUEUE_SIZE,HEARTBEAT_DEVICE,LOG_BACKUP_COUNT,CAMERA_HEARTBEAT_TIMEOUT,BUILD_ID
# from Utils.utils import create_logger


class RabbitmqManager:
     
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self,rabbitmq_reader_queue,shared_camera_dict,shared_device_dict):
        try: 
            self.config_manager = ConfigManager().get_instance()
            self.logger = CustomLogger("RabbitmqManager").get_logger(
            log_file="logs/rabbitmq_manager.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True),)
            self.rabbitmq_reader_queue = rabbitmq_reader_queue
            self.shared_camera_dict = shared_camera_dict
            self.shared_device_dict = shared_device_dict
            self.rabbitmq_username = self.config_manager.get("RABBITMQ_USERNAME", "guest")
            self.rabbitmq_password = self.config_manager.get("RABBITMQ_PASSWORD", "guest")
            self.rabbitmq_host = self.config_manager.get("RABBITMQ_HOST", "localhost")
            self.rabbitmq_port = self.config_manager.get("RABBITMQ_PORT", 5672)
            self.BUID_ID = self.config_manager.get("BUILD_VERSION","v1")
            self.rabbitmq_queue_name = self.config_manager.get("RABBITMQ_QUEUE","gotilo_queue")
            self.rabbitmq_exchange = self.config_manager.get("RABBITMQ_EXCHANGE","gotilo_exchange")
            self.rabbitmq_routing_key = self.config_manager.get("RABBITMQ_ROUTING_KEY","gotilo_routing_key")
            self.is_running = True
            self.last_heartbeat_sent_time = None
            self.connected = False
            self.retry_count = 0
            self.credentials = pika.PlainCredentials(self.rabbitmq_username,self.rabbitmq_password)
            self.parameters = pika.ConnectionParameters(
                            host=self.rabbitmq_host,port=self.rabbitmq_port,virtual_host="/",credentials=self.credentials,heartbeat=60)
            self.reconnect()
            self.sender_thread = Thread(target=self.send_message,daemon=True)

        except Exception as exec:
            print("RabbitmqManager __init__ error:",exec)
            traceback.print_exc()
    

    def reconnect(self, close_flag=False):
        self.connected = False
        self.retry_count = 0
        while not self.connected:
            time.sleep(1)
            try:
                if close_flag:
                    if not self.channel.is_closed or self.channel.is_open:
                        self.channel.close()
                    if not self.connection.is_closed or self.connection.is_open:
                        self.connection.close()
                    
                self.connection = pika.BlockingConnection(self.parameters)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.rabbitmq_queue_name, durable=True)
                self.channel.exchange_declare(exchange=self.rabbitmq_exchange, exchange_type='direct', durable=True)
                self.channel.confirm_delivery()
                self.connected = True
            except Exception:
                self.retry_count += 1
                time.sleep(5)
                self.logger.error(f"Exception in reconnect {traceback.format_exc()}")
    

    def send_message(self):
        try:
            self.logger.info("started rabbitmq sending thread")
            while self.is_running:
                try:
                    if self.rabbitmq_reader_queue.empty():
                        time.sleep(0.1)
                        continue
                    message = self.rabbitmq_reader_queue.get()
                    
                    if message is None:
                        time.sleep(0.1)
                        continue
                    self.logger.info(f"{message=}")
                    message = json.dumps(message)
                    self.logger.info("message dumped")
                    self.logger.info(f"publishing message")
                    self.basic_publish(message)
                    
                        


                
                except Exception as exec:
                    self.logger.error(f"Exception in send_message inner {traceback.format_exc()}")
        
        except Exception as exec:
            self.logger.error(f"Exception in send_message {traceback.format_exc()}")
            self.reconnect(close_flag=True)
    
    def basic_publish(self,message):
        try:
            while True:
                try:
                    # if not self.connected:
                    #     self.reconnect(close_flag=True)
                    self.logger.info(f"publishing message")

                    self.channel.basic_publish(
                        exchange=self.rabbitmq_exchange,
                        routing_key=self.rabbitmq_routing_key,
                        body=message,
                        properties=pika.BasicProperties(
                            delivery_mode=2,  # Make message persistent
                        )
                    
                    )
                    self.logger.info(f"published message")

                    break
                except Exception as e:
                    self.reconnect()
                    self.logger.error(f"Error {e}")
            return True
        except Exception as exec:
            self.logger.error(f"Exception in basic_publish {traceback.format_exc()}")
            self.reconnect(close_flag=True)
            return False
    
    def start(self):
        try:
            self.is_running = True
            self.sender_thread.start()
            self.logger.info("RabbitmqManager started successfully.")
        except Exception as exec:
            self.logger.error(f"Exception in start {traceback.format_exc()}")
               
                

