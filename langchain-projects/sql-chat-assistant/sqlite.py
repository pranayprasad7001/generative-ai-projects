import sqlite3

# Connect to Sqlite Database
connection = sqlite3.connect("student.db")

# Create Cursor Object to insert, update, delete and fetch data
cursor_obj = connection.cursor()

# Create Table
table_info ="""
CREATE TABLE IF NOT EXISTS STUDENT(
    ID INTEGER PRIMARY KEY,
    NAME VARCHAR(50) NOT NULL,
    CLASS VARCHAR(25) NOT NULL,
    SECTION CHAR(1) NOT NULL,
    MARKS INTEGER
)
"""

# Create table using execute() method
cursor_obj.execute(table_info)

# Put your data into a list of tuples (leaving out ID so it auto-increments)
students_data = [
    ('Pranay', 'Machine Learning', 'A', 90),
    ('Krish', 'Artificial Intelligence', 'B', 83),
    ('Harsh', 'Data Science', 'A', 86),
    ('Aditya', 'Automobile', 'C', 74)
]

# Use executemany to insert all of them at once
cursor_obj.executemany('''
    INSERT INTO STUDENT (NAME, CLASS, SECTION, MARKS) 
    VALUES(?, ?, ?, ?)
''', students_data)

# Display all the records
print("The inserted records are:")
data = cursor_obj.execute("SELECT * FROM STUDENT")
for row in data:
    print(row)

# Commit changes
connection.commit()

# Close connection
connection.close()