from . import db
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))

class TECParams(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    year = db.Column(db.Integer, nullable=False)
    day_of_year = db.Column(db.Integer, nullable=False)
    hour_of_day = db.Column(db.Integer, nullable=False)
    rz_12 = db.Column(db.Integer, nullable=False)
    ig_12 = db.Column(db.Integer, nullable=False)
    ap_index = db.Column(db.Float, nullable=False)
    kp_index = db.Column(db.Float, nullable=False)
    tec_output = db.Column(db.Float, nullable=True)