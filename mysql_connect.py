import pymysql

class mysql_connect():
    def __init__(self, user = 'admin', password = "better1234",
    host='database-1.cyooqkxaxvqu.us-east-1.rds.amazonaws.com',
    port = 3306, database = 'mysql'):

        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.log = ""
        try:
            self.db = pymysql.connect(host = self.host, port = self.port,
            user = self.user, passwd = self.password,
            db = self.database, charset='utf8')
            self.connect = True
        except Exception as e:
            self.connect = False
            self.log = e
        self.cursor = self.db.cursor()

    def __del__(self):
        self.db.close()

    def querry(self, sql):
        try:
            with self.cursor as cursor:
                cursor.execute(sql)
                return True, cursor
        except Exception as e:
            return False, e

    def get_items_list(self, sql):
        ch, cursor = self.querry(sql)
        if ch:
            fetch = cursor.fetchone()
            result = [fetch]
            while fetch:
                fetch = cursor.fetchone()
                result.append(fetch)
            return result
        else:
            return cursor

if __name__ == "__main__":
    conn = mysql_connect(
    user = 'admin',
    password = "better1234",
    host='database-1.cyooqkxaxvqu.us-east-1.rds.amazonaws.com',
    port = 3306,
    database = 'better',
    )
    if conn.connect:
        print(conn.get_items_list('show tables'))
