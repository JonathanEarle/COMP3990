#Generate a random db (Add in known plate to test for positive ID)
import redis
import sys
import dbOperations as db
sys.path.insert(0,'../CNN')
import GenPlates as gp

def populateVehicles(r):
	for i in range(50):
		plate=gp.generateCode().replace(" ","").replace("\n","")
		r.sadd("vehicles",*set([plate]))

def populateBlacklist(r):
	for i in range(10):
		r.sadd("blacklist",*set([r.srandmember("vehicles")]))

def main():
	r=db.setupDB()
	r.flushall()

	r.sadd("vehicles",*set(["PCX172"]))
	r.sadd("blacklist",*set(["PCX172"]))
	
	populateVehicles(r)
	populateBlacklist(r)

	print(r.smembers("vehicles"))
	print(r.smembers("blacklist"))

if __name__=="__main__":
	main()