from flask import Flask, render_template, request,json,jsonify

import cv2
import glob
import numpy as np
import sqlite3
from sqlite3 import Error
import random 
# class user:
#   def __init__(my, name):
#     my.name = name

#   def myfunc(abc):

#     print("Hello my name is " + abc.name)
#     return abc.name

# check = "inin"
# t = ""
# def sett(s):
# 	t = str(s)
# def gett():
# 	return t

x = []
x1 = ['/static/1/0_1.jpg','/static/1/0_2.png','/static/1/0_3.png','/static/1/0_4.jpg','/static/1/0_5.jpg','/static/1/0_6.jpg','/static/1/0_7.jpg','/static/1/0_8.jpg','/static/1/0_9.jpg','/static/1/0_10.png'] 
x2 = ['/static/2/2_1.jpg','/static/2/2_2.jpg','/static/2/2_3.jpg','/static/2/2_4.jpg','/static/2/2_5.jpg','/static/2/2_6.jpg','/static/2/2_7.jpg','/static/2/2_8.jpg','/static/2/2_9.jpg','/static/2/2_10.jpg'] 
num = ['/static/no/1.png','/static/no/2.png','/static/no/3.png','/static/no/4.png','/static/no/5.png','/static/no/6.png','/static/no/7.png','/static/no/8.png','/static/no/9.png','/static/no/10.png',]

fin = []    
fin1 = []

count = []
for i in range(0,len(x1)):
	r1 = random.randint(0, 1) 
	if(r1==0):
		fin.append(x1[i])
		fin.append(x2[i])
		count.append(0)
		count.append(1)

	if(r1==1):
		fin.append(x2[i])
		fin.append(x1[i])
		count.append(1)
		count.append(0)

print(fin)	
odd = []
even = []

for i in range(0,len(fin)):
	if(i%2==0):
		even.append(fin[i])

	else:
		odd.append(fin[i])	




app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template("new.html")
@app.route('/dis')
def my():
	# fin = []    
	# fin1 = []

	# count = []
	# for i in range(0,len(x1)):
	# 	r1 = random.randint(0, 1) 
	# 	if(r1==0):
	# 		fin.append(x1[i])
	# 		fin.append(x2[i])
	# 		count.append(0)
	# 		count.append(1)

	# 	if(r1==1):
	# 		fin.append(x2[i])
	# 		fin.append(x1[i])
	# 		count.append(1)
	# 		count.append(0)
	# print(fin)	
	# odd = []
	# even = []

	# for i in range(0,len(fin)):
	# 	if(i%2==0):
	# 		even.append(fin[i])

	# 	else:
	# 		odd.append(fin[i])	

	return render_template("index.html", users=odd,fusers = even,kusers = num)    


@app.route('/result', methods=['GET', 'POST'])
def index():
	

	if request.method == 'POST':
		a = []
		text = request.form['text']
		# print("@@@@@@"+text+"@@@@@@")
		c = 0
		if request.form.get(fin[0]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[1]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[2]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[3]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[4]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[5]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[6]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[7]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[8]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[9]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[10]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[11]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[12]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[13]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[14]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[15]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[16]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[17]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[18]):
			a.append(1)
		else:
			a.append(0)
		if request.form.get(fin[19]):
			a.append(1)
		else:
			a.append(0)		
	print(a)
	for i in range(0,len(a)):
		if(a[i]!=count[i]):
			c+=1
	print(c)		
	c = int(c)
	try:
		conn = sqlite3.connect("db location")
		print(conn)
	except Error as e:
		print(e)
	cur = conn.cursor()	
	res = str(text)
	with conn:
		print(res,c/2)
		project = (res,c/2)		
		cur.execute(''' INSERT INTO projects(name,score) VALUES(?,?) ''', project)	
	cur = conn.cursor()
	cur.execute("SELECT * FROM projects order by score desc")
	data = cur.fetchall()
	#return jsonify(c)
	return render_template('temp.html', data=data)		   
if __name__ == '__main__':
    app.run(port=5000,host="0.0.0.0", debug=True)
