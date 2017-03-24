#Generate a random db (Add in known plate to test for positive ID)
import redis
import GenPlates as gp

def setupDB():
	return redis.Redis(
    host='localhost',
    port=6379, 
    password='')

def populateVehicles(r):
	for i in range(50):
		plate=gp.generateCode().replace(" ","").replace("\n","")
		r.sadd("vehicles",*set([plate]))

def populateBlacklist(r):
	for i in range(10):
		r.sadd("blacklist",*set([r.srandmember("vehicles")]))

def main():
	r=setupDB()
	r.delete("vehicles")
	r.delete("blacklist")

	r.sadd("vehicles",*set(["PCX172"]))
	r.sadd("blacklist",*set(["PCX172"]))
	
	populateVehicles(r)
	populateBlacklist(r)

	print(r.smembers("vehicles"))
	print(r.smembers("blacklist"))

if __name__=="__main__":
	main()