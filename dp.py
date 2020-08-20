import sqlite3
from sqlite3 import Error
 
 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
 
    return conn
 
 
def create_project(conn, project):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO projects(name,score)
              VALUES(?,?) '''
    cur = conn.cursor()
    cur.execute(sql, project)
    return cur.lastrowid
 
 

 
 
def main():
    database = r"login.db"
 
    # create a database connection
    conn = create_connection(database)
    with conn:
        # create a new project
        project = ('john',10);
        project_id = create_project(conn, project)
 

 
if __name__ == '__main__':
    main()

