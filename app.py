from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from collections import Counter
import os

app = Flask(__name__)
app.secret_key = "super_secret_key_change_me"

# -------------------------
# Database Configuration (SQLite)
# -------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(BASE_DIR, "evm.db")

app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# -------------------------
# Voter Model
# -------------------------
class Voter(db.Model):
    __tablename__ = "voters"

    id = db.Column(db.Integer, primary_key=True)
    voter_id = db.Column(db.String(50), unique=True, nullable=False)  # from CV/hardware
    name = db.Column(db.String(100), nullable=False)
    vote_status = db.Column(db.Boolean, default=False, nullable=False)
    vote_party = db.Column(db.String(100), nullable=True)  # NULL initially

    def __repr__(self):
        return f"<Voter {self.voter_id} - {self.name}>"


# -------------------------
# Admin Login Config
# -------------------------
ADMIN_USER_ID = "admin"
ADMIN_PASSWORD = "1234"


def is_logged_in():
    return session.get("logged_in") is True


# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    if is_logged_in():
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")

        if user_id == ADMIN_USER_ID and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["user_id"] = user_id
            flash("Login successful! Welcome, Admin.", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials. Please try again.", "danger")

    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    if not is_logged_in():
        flash("Please login to access the dashboard.", "warning")
        return redirect(url_for("login"))

    voters = Voter.query.order_by(Voter.voter_id).all()

    # Only count votes where vote_status is True and vote_party is not null
    party_votes = Counter(
        v.vote_party for v in voters if v.vote_status and v.vote_party
    )

    ranked_parties = sorted(party_votes.items(), key=lambda x: x[1], reverse=True)

    if ranked_parties:
        winning_party = ranked_parties[0][0]
        winning_votes = ranked_parties[0][1]
    else:
        winning_party = None
        winning_votes = 0

    total_votes = sum(1 for v in voters if v.vote_status)

    return render_template(
        "dashboard.html",
        voters=voters,
        party_votes=party_votes,
        ranked_parties=ranked_parties,
        winning_party=winning_party,
        winning_votes=winning_votes,
        total_votes=total_votes,
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0',debug=True)
