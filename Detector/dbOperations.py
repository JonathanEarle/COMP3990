#Updates the database with all the detected plates once there is a connection
import time
import socket
import redis
REMOTE_SERVER = "www.google.com"

def setupDB():
	return redis.Redis(
    host='localhost',
    port=6379, 
    password='')

def isConnected():
	try:
		host = socket.gethostbyname(REMOTE_SERVER)
		s = socket.create_connection((host, 80), 2)
		return True
	except:
		pass
	return False

def update(r):
	while True:
		if(isConnected()):
			with open("detected.csv","rb") as det:
				for line in det:
					elements=line.split(",")
					code=elements[0]
					t=elements[1].replace("\n","")#Time detected
					r.sadd("detected:"+code,*set([str(t)]))
			open("detected.csv", 'wb').close()
		time.sleep(10)

def main():
	redisDb=setupDB()
	update(redisDb)

if __name__=="__main__":
	main()