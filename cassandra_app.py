from cassandra.cluster import Cluster

KEYSPACE = "pictureResult"

def insertResult(timeStamp, filename, result):

    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)

    session = cluster.connect()
    session.execute("USE pictureResult")
    session.execute("INSERT INTO Result (timeStamp, fileName, result) VALUES (%s, %s, %s)", (timeStamp, filename, result))
    return 0


