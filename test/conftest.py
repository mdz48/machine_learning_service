"""Fixtures de test que aíslan la BD real.

Cada test corre dentro de una transacción externa con SAVEPOINTs anidados: aunque el
endpoint `/predict` llame `db.commit()` internamente, todo se revierte al terminar el
test, así que la base compartida NO queda con inferencias basura.

Requisito: debe existir un `MLModel` activo (sembrado por `seed_model.py`); `main.py` lee
su id al importar y las inserciones de inferencias lo referencian por FK.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, get_db
from database import engine


@pytest.fixture()
def db_session():
    connection = engine.connect()
    trans = connection.begin()
    # join_transaction_mode="create_savepoint" (SQLAlchemy 2.0) reinicia el SAVEPOINT
    # tras cada commit interno del endpoint, sin cerrar la transacción externa.
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        trans.rollback()
        connection.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
