# # db_init.py
# from app import db, Voter, app

# with app.app_context():
#     db.drop_all()
#     db.create_all()

#     sample_voters = [
#         {"voter_id": "V001", "name": "Md. Munsur Ali"},
#         {"voter_id": "V002", "name": "Sabbir Hossain Talukder"},
#         {"voter_id": "V003", "name": "Sheikh Latifur Rahman"},
#         {"voter_id": "V004", "name": "Md. Minhazul Islam"},
#         {"voter_id": "V005", "name": "Sarafat Hussain Abhi"},
#     ]

#     voters = []
#     for v in sample_voters:
#         voters.append(
#             Voter(
#                 voter_id=v["voter_id"],
#                 name=v["name"],
#                 vote_status=False,
#                 vote_party=None,
#             )
#         )

#     db.session.add_all(voters)
#     db.session.commit()

#     print("Database initialized and sample voters inserted.")


####### New
# db_init.py
from app import db, Voter, app

with app.app_context():
    db.drop_all()
    db.create_all()

    sample_voters = [
        {"voter_id": "V001", "name": "Md. Munsur Ali"},
        {"voter_id": "V002", "name": "Sabbir Hossain Talukder"},
        {"voter_id": "V003", "name": "Sheikh Latifur Rahman"},
        {"voter_id": "V004", "name": "Md. Minhazul Islam"},
        {"voter_id": "V005", "name": "Sarafat Hussain Abhi"},
        {"voter_id": "V006", "name": "Sheikh Mohammad Aiyan"},
        {"voter_id": "V007", "name": "Mohammad Shakhawat Hossain"},
        {"voter_id": "V008", "name": "Hasin Hasnat"},
        {"voter_id": "V009", "name": "Abdul Momin"},
        {"voter_id": "V010", "name": "Ehasanul Haque Efti"},
        {"voter_id": "V011", "name": "Robiul Pramanik"},
        {"voter_id": "V012", "name": "Raynul Islam Rafin"},
    ]

    voters = []
    for v in sample_voters:
        voters.append(
            Voter(
                voter_id=v["voter_id"],
                name=v["name"],
                vote_status=False,
                vote_party=None,
            )
        )

    db.session.add_all(voters)
    db.session.commit()

    print("Database initialized and voters inserted.")

