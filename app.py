from werkzeug.utils import secure_filename
from flask import Flask, request, abort, render_template
import csv

def assemble(prod, state):
    result = None
    
    with open('submission.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
           
            id_raw = row[0]
            split = id_raw.split("_")
            p = split[0] + "_" + split[1] + "_" + split[2]
            s = split[3] + "_" + split[4]

            if split[5] == "validation":
                continue
            
            if p == prod and s == state:
                result = row[1:]
                break
    return result

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    p=request.form.get("product")
    s=request.form.get("state") 

    x = assemble(p, s)

    return render_template("evaluation.html", x=x, item=p, store=s)

if __name__ == "__main__":
    app.run()
