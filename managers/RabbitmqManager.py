import time
import pika
import json
import traceback
import jsons
import base64
import cv2
from threading import Thread
from queue import Queue
from datetime import datetime, timedelta,timezone
from Modules.CustomLogger import CustomLogger
from managers.ConfigManager import ConfigManager
# from constant import HIDE_SETTINGS,DEBUG_MODE,RABBITMQ_USERNAME,RABBITMQ_PASSWORD,RABBITMQ_HOST,RABBITMQ_PORT,RABBITMQ_QUEUE_SIZE,HEARTBEAT_DEVICE,LOG_BACKUP_COUNT,CAMERA_HEARTBEAT_TIMEOUT,BUILD_ID
# from Utils.utils import create_logger
import threading


class RabbitMQConnection:
    _instance = None
    _lock = threading.Lock()   # protects reconnection

    @classmethod
    def reset_instance(cls):
        with cls._lock:
            cls._instance = None

    def __new__(cls, params, logger):
        if cls._instance is None:
            with cls._lock:  # ensure singleton creation is thread-safe
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.params = params
                    cls._instance.logger = logger
                    cls._instance._connect()
        return cls._instance

    def _connect(self):
        while True:
            try:
                self.connection = pika.BlockingConnection(self.params)
                self.channel = self.connection.channel()
                self.logger.warning("RabbitMQConnection: Connected")
                break
            except Exception as e:
                self.logger.error(f"RabbitMQConnection: Connection failed, retrying... {e}")
                time.sleep(5)

    def get_channel(self):
        with self._lock:  # only one thread at a time checks/reconnects
            try:
                if self.connection.is_closed or self.channel.is_closed:
                    self.logger.warning("RabbitMQConnection: Reconnecting because channel/connection is closed")
                    self._connect()
            except Exception:
                self.logger.warning("RabbitMQConnection: Exception, forcing reconnect")
                self._connect()
            return self.channel
        

class RabbitMQManager():
    def __init__(self,rabbitmq_queue,shared_camera_dict,shared_device_dict):
        self.config_manager = ConfigManager.get_instance()
        self.logger = CustomLogger("RabbitMQManager").get_logger(
            log_file="logs/rabbitmq_manager.log",
            log_level=self.config_manager.get("LOG_LEVEL", 10),
            log_to_console=self.config_manager.get("LOG_TO_CONSOLE", True),
        )

    
        self.rabbitmq_username = self.config_manager.get("RABBITMQ_USERNAME","guest")
        self.rabbitmq_password = self.config_manager.get("RABBITMQ_PASSWORD","guest")
        self.rabbitmq_host = self.config_manager.get("RABBITMQ_HOST","localhost")
        self.rabbitmq_port = self.config_manager.get("RABBITMQ_PORT",5672)
        self.BUID_ID = self.config_manager.get("BUILD_VERSION","v1")
        self.rabbitmq_queue_name = self.config_manager.get("RABBITMQ_QUEUE_NAME","gotilo_queue")

        self.is_running = True
        # rabbitmq_username = self.rabbitmq_username
        # rabbitmq_password = self.rabbitmq_password
        # rabbitmq_host = self.rabbitmq_host  
        # rabbitmq_port = self.rabbitmq_port
        self.rabbitmq_queue = rabbitmq_queue

        self.backup_queue = Queue(self.config_manager.get("RABBITMQ_QUEUE_SIZE",500))
        self.shared_camera_dict = shared_camera_dict
        self.shared_device_dict=shared_device_dict

        self.exchange_name = "amq.topic"
        self.params = pika.ConnectionParameters(host=self.rabbitmq_host, port=self.rabbitmq_port, credentials=pika.PlainCredentials(self.rabbitmq_username, self.rabbitmq_password),heartbeat=60)
        self.logger.warning(f"Connecting to RabbitMQ server")

        self.connection = None
        self.channels = {}
        self.sending_queue = "gotilo_dummy_queue"
        self.reconencting_sending_queue(self.sending_queue)
        self.logger.warning(f"Connected to RabbitMQ server")

        self.last_heartbeat_sent_time = None
        self.heartbeat_device = self.config_manager.get("HEARTBEAT_DEVICE",True)
        self.camera_heartbeat_interval = self.config_manager.get("CAMERA_HEARTBEAT_TIMEOUT",20)
        
    
    def start(self):
        Thread(target=self.check_thread_and_queues_status).start()
        # Thread(target=self.check_camera_status).start()
        Thread(target=self.send_message).start()
        Thread(target=self.receive_message).start()
        
    def check_thread_and_queues_status(self):
        while True:
            # print("sleeping for 10 seconds in check_thread_and_queues_status")
            # if DEBUG_MODE:
            # self.logger.warning(f"The size of rabbitmq queue is {self.rabbitmq_queue.qsize()}")
            time.sleep(10)
    
    def send_to_queue(self,queue_name,message_type, message):
        if self.rabbitmq_queue.full():
            self.logger.error("RabbitMQ queue is full, adding message to backup queue")
        else:
            message= {"queue_name":queue_name,"message_type":message_type,"message":message}
            self.rabbitmq_queue.put(message)
            # if DEBUG_MODE:
            # self.logger.warning(f"Added a message into the rabbit mq QUEUE")

    def reconencting_sending_queue(self,queue_name):
        self.logger.warning("Inside the reconnect and publish")
        connected = False
        while not connected:
            print("sleeping for 5 seconds in reconnect and publish")
            self.logger.info(f"Sleeping for 5 seconds in the reconnect and publish")
            time.sleep(5)
            try:
                self.logger.warning("Reconnecting")
                self.logger.warning(f"Inside the try of reconnect and publish")
                self.logger.warning(f"The size of the queue is {self.rabbitmq_queue.qsize()}")
                conn = RabbitMQConnection(self.params, self.logger)
                self.connection = conn.connection
                
                if not self.connection or self.connection.is_closed:
                    self.logger.warning("Connection is Closed.")
                    # RabbitMQConnection._instance=None
                    RabbitMQConnection.reset_instance()
                    conn = RabbitMQConnection(self.params, self.logger)
                    self.connection = conn.connection
                    self.logger.warning("New Connection created.")
                
                if queue_name in self.channels.keys():
                    self.logger.warning(f"The queue {queue_name} is already present in the channels")
                    channel = self.channels[queue_name]
                    # close the channel if it is already open
                    if channel.is_open:
                        self.logger.warning(f"The channel {queue_name} is already open, closing it")
                        channel.close()
                        self.logger.warning(f"The channel {queue_name} is closed")
                          
                # self.channel.queue_declare(queue=self.sending_queue, durable=True)
                channel = self.connection.channel()
                self.logger.warning(f"Created a new channel for the queue {queue_name}")
                # Declare the exchange and queue
                channel.exchange_declare(exchange=queue_name, exchange_type='direct', durable=True)
                channel.queue_declare(queue=queue_name, durable=True)
                channel.queue_bind(exchange=queue_name, queue=queue_name, routing_key=queue_name)
                channel.confirm_delivery()
                self.logger.warning(f"Connected to the channel {queue_name}")
                self.channels[queue_name] = channel
                
                connected = True
            except Exception as e:
                print("Error in reconnecting and publishing")
                self.logger.error(f"Inside the exception of the reconnect and publish, the error is {e} {traceback.format_exc()}")
                self.logger.error(f"Sleeping for 10 seconds")
                time.sleep(10)

    def basic_publish(self,queue_name,message,message_type):
        # if DEBUG_MODE:
        #     self.logger.info(f"Sending the message for the channel basic publish")
        while True:
            try:
                self.logger.info(f"Inside the basic publish")
                self.logger.info(f"Message type : {message_type}"   )
                self.logger.info(f"Queue name : {queue_name}"   )
                # channel = self.channels[queue_name]
                channel = self.channels.get(queue_name)
                if channel is None:
                    self.logger.error(f"Channel for queue {queue_name} is not found, reconnecting")
                    self.reconencting_sending_queue(queue_name)
                    channel = self.channels[queue_name]
                
                channel.basic_publish(exchange=queue_name, routing_key=queue_name, body=message, properties=pika.BasicProperties(delivery_mode=2))

                self.logger.warning(f"Published the message of type {message_type}")
                
                # self.logger.warning(f"Exiting the basic publish")
                break

            except Exception as e:
                self.logger.error(f"Exception while publishing message, {e}",exc_info=True)
                self.reconencting_sending_queue(queue_name)

    

    def change_index_callback(self,ch, method, properties, body):
        try:
            body = body.decode("utf-8")
            body = json.loads(body)
            message_type = body["message_type"]
            # device_id_ = body["device_id"]
            camera_id_ = body["camera_id"]
            self.logger.info(f"Received the message {message_type}")
            self.logger.info(f"Camera id : {camera_id_}")

            

            if body["message_type"] == "capture_image":
                self.logger.warning(f"Received the message to capture the image")
                
                
                if camera_id_ not in self.shared_camera_dict.keys():
                    self.logger.critical(f"The camera id is not matching {camera_id_} and {self.shared_camera_dict.keys()}")
                    return
                
                if "image" not in self.shared_camera_dict[str(camera_id_)].keys():
                    self.logger.critical(f"No image found for the camera id {camera_id_}")
                    return
                
                frame = self.shared_camera_dict[str(camera_id_)]["image"]
                if frame is None:
                    self.logger.critical(f"No image found for the camera id {camera_id_}")
                    return
                frame = cv2.imencode('.jpg', frame)[1].tobytes()
                base64_image = base64.b64encode(frame).decode("utf-8")


                queue_name = str(self.shared_camera_dict[str(camera_id_)]["queue_name"])
                device_id = str(self.shared_camera_dict[str(camera_id_)]["device_id"])
                site_id = str(self.shared_camera_dict[str(camera_id_)]["site_id"])
                site_name = str(self.shared_camera_dict[str(camera_id_)]["site_name"])


                self.send_to_queue(queue_name,"update_camera_image",{"device_id":device_id,"site_id":site_id,"site_name":str(site_name),"camera_id":camera_id_,"image_base64":base64_image})
                self.logger.warning(f"Sent the camera image for camera id {camera_id_}")
                
            
            if body["message_type"] =="reload":
                
                
                if camera_id_ not in self.shared_camera_dict.keys():
                    self.logger.critical(f"The camera id is not matching {camera_id_} and {self.shared_camera_dict.keys()}")
                    return
                
                if "reload" not in self.shared_camera_dict[str(camera_id_)].keys():
                    self.logger.critical(f"No reload found for the camera id {camera_id_}")
                    return
                
                if not self.shared_camera_dict[str(camera_id_)]["reload"]:
                    self.logger.info(f"Reloading the camera {camera_id_}")
                    last_active_time = self.shared_camera_dict[str(camera_id_)]["last_active_time"] 
                    image = self.shared_camera_dict[str(camera_id_)]["image"]
                    reload = True


                    self.shared_camera_dict[str(camera_id_)]["last_active_time"] = last_active_time
                    self.shared_camera_dict[str(camera_id_)]["image"]= image
                    self.shared_camera_dict[str(camera_id_)]["reload"]= reload
                    self.shared_camera_dict[str(camera_id_)]["fps"]= 0
                    
                    # self.shared_camera_dict[str(camera_id_)]={
                    #     "last_active_time":last_active_time,
                    #     "image":image,
                    #     "reload":reload,
                    #     "fps":0
                    # }


        except Exception as e:
            self.logger.error(f"Exception in change index callback, {e}",exc_info=True)
  
    def receive_message(self):
        self.logger.warning(f"Started the receive message thread")
        connected = False
        while not connected:
            print("sleeping for 5 seconds in receive message")
            self.logger.warning(f"Sleeping for 5 seconds in the receive message")
            time.sleep(5)
            try:

                self.receive_connection = pika.BlockingConnection(self.params)
                self.receiving_channel = self.receive_connection.channel()
                self.logger.warning(f"Connected to RabbitMQ server for receiving messages")
                connected = True
            except Exception as e:
                print("Error in connecting to RabbitMQ server for receiving messages")
                self.logger.error(f"Error in connecting to RabbitMQ server for receiving messages, {e}")
                time.sleep(10)

        while True:
            # print("Sleeping for 5 seconds in receive message")
            time.sleep(5)
            try:
                # if site_id is None:
                #     continue
                self.receiving_queue_ = self.receiving_channel.queue_declare('', exclusive=True).method.queue
                # self.logger.warning(self.receiving_queue_)
                self.receiving_channel.queue_bind(exchange=self.exchange_name, queue=self.receiving_queue_, routing_key=str(self.BUID_ID)+".#")
                
                # self.logger.info(f"Waiting for the message")
                self.receiving_channel.basic_consume(queue=self.receiving_queue_, on_message_callback=self.change_index_callback, auto_ack=True)
                self.receiving_channel.start_consuming()
                
            except Exception as e:
                print("Exception in receving message : ",e)
                self.logger.error(f"Exception in receving message : {e}")
                self.receive_connection = pika.BlockingConnection(self.params)
                self.receiving_channel = self.receive_connection.channel()
                self.receiving_queue_ =self.receiving_channel.queue_declare('', exclusive=True).method.queue
                
                
            finally:
                time.sleep(1)
                continue
        
    def check_camera_status(self):
        self.logger.warning(f"Started the check camer-a status thread")
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                for camera_id in self.shared_camera_dict.keys():
                    camera_id = str(camera_id)
                    current_camera = self.shared_camera_dict[camera_id]

                    if "queue_name" not in current_camera:
                        self.logger.warning(f"No queue_name found for camera {camera_id}, using default queue {self.sending_queue}")
                        queue_name = self.sending_queue  # Use default queue
                    else:
                        queue_name = str(current_camera["queue_name"])

                    
                    # for device_id in self.shared_device_dict.keys():
                    #     device = self.shared_device_dict[device_id]
                    #     device_id = str(device["device_id"])
                    #     device_name = str(device["device_name"])
                    #     site_id = str(device["site_id"])
                    #     site_name = str(device["site_name"])
                    #     queue_name = str(device["queue_name"])


                    if "last_active_time" not in current_camera.keys() or "image" not in current_camera.keys() or current_camera["last_active_time"] is None:
                        continue
                    if (current_time - current_camera["last_active_time"]).total_seconds() < 30:
                        # the camera is online and send the hearbeat for this camera
                        self.logger.warning(f"Camera id {camera_id} heartbeat sent")
                        
                        data = {"camera_id":camera_id,"timestamp":current_time,"site_id":current_camera["site_id"],"fps":current_camera["fps"]}

                        self.send_to_queue(queue_name,"heartbeat",data)
                
                if self.heartbeat_device:
                    
                    for device_id in self.shared_device_dict.keys():
                        device = self.shared_device_dict[device_id]
                        device_id = str(device["device_id"])
                        device_name = str(device["device_name"])
                        site_id = str(device["site_id"])
                        site_name = str(device["site_name"])
                        queue_name = str(device["queue_name"])

                        message = {"site_id":site_id,"device_id":device_id,"current_time":current_time}

                        self.send_to_queue(queue_name,"heartbeat_device",message)
                        self.logger.warning(f"Heartbeat Device message sent")
            except Exception as e:
                self.logger.error(f"Exception in check_camera_status : {e}",exc_info=True)
            time.sleep(self.camera_heartbeat_interval)


    def send_message(self):
        self.logger.info(f"Started the send message thread")
        while True:
            current_queue = self.rabbitmq_queue
            if not self.backup_queue.empty():
                self.logger.warning(f"Backup queue is not empty, will use the backup queue")
                current_queue = self.backup_queue
            
            if not current_queue.empty():
                try:
                    
                    # print("total message in queue",current_queue.qsize())
                    # self.logger.info(f"total message in queue: {current_queue.qsize()}")
                    
                    raw_message = current_queue.get()
 
                    message_type = raw_message["message_type"]

                    queue_name = raw_message["queue_name"]
                    
                    self.logger.warning(f"Message type is {message_type} and queue name is {queue_name}")


                    # if DEBUG_MODE:
                    #     self.logger.info(f"the size of queue after accessing it {self.rabbitmq_queue.qsize()}")
                    
                    message = jsons.dumps(raw_message)
                    # if DEBUG_MODE:
                    #     self.logger.info(f"Converted the message into json, and will now publish in self.basic_publish")
                    
                    self.basic_publish(queue_name,message,message_type)
                except Exception as e:
                    self.logger.error(f"Exception in send message : {e}",exc_info=True)
                    print(traceback.print_exc())
                    self.backup_queue.put(raw_message)
                    # self.logger("Sleeping for 1 second in send message except")
                    time.sleep(1)
                finally:
                    # print("Inside the finally of send message")
                    continue
            else:
                time.sleep(5)
                continue    
                
                

