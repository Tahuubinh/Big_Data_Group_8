#importing the module 
import logging 

#now we will Create and configure logger 
logging.basicConfig(filename="std.log", 
					format='%(asctime)s %(message)s', 
					filemode='w') 

#Let us Create an object 
logger=logging.getLogger() 

#Now we are going to Set the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

#some messages to test
logger.debug("This is just a harmless debug message") 
logger.debug("This is just a harmless rewr message") 
logger.debug("This is just a harmless dsfsd message") 