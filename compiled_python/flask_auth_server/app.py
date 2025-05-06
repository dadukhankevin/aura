# Standard Python Imports
# Use render_template_string for simplicity
from flask import Flask, request, session, redirect, url_for, render_template_string
import os

# Built-in imports (added by compiler)
import os
import sys
import re
from typing import List, Dict, Any, Optional, TypeAlias

# Type Aliases from 'desc' blocks (for clarity)
# A secure secret key for session management (use os.urandom)
secret_key: TypeAlias = Any
login_route: TypeAlias = Any  # /login
logout_route: TypeAlias = Any  # /logout
protected_route: TypeAlias = Any  # /protected
home_route: TypeAlias = Any  # /
# A simple dictionary representing the allowed user {'username': 'test', 'password': 'pw'}
mock_user: TypeAlias = Dict

# --- Aura Block: aura_function:create_app ---


def create_app() -> Flask:
    """Creates and configures the Flask application instance."""
    app = Flask(__name__)
    app.secret_key = os.urandom(24)
    return app

# --- Aura Block: aura_function:login_handler ---


def login_handler():
    """Handles POST requests to /login. Checks mock credentials."""
    mock_user = {'username': 'test', 'password': 'pw'}
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == mock_user['username'] and password == mock_user['password']:
            session['logged_in'] = True
            return redirect('/protected')
        else:
            return "Invalid Credentials", 401
    else:
        return render_template_string('<form method="POST"><input type="text" name="username"/><input type="password" name="password"/><input type="submit"/></form>')

# --- Aura Block: aura_function:render_login_form ---


def render_login_form():
    html = '''<form method="POST" action="/login"><label for="username">Username:</label><br><input type="text" id="username" name="username"><br><label for="password">Password:</label><br><input type="password" id="password" name="password"><br><input type="submit" value="Submit"></form>'''
    return html

# --- Aura Block: aura_function:logout_handler ---


def logout_handler():
    session['logged_in'] = False
    return redirect('/')

# --- Aura Block: aura_function:protected_handler ---


def protected_handler():
    """Handles requests to /protected. Requires login."""
    if session.get('logged_in') is True:
    return "Welcome to the protected area!"
    else:
    return redirect(url_for('login_route'))

# --- Aura Block: aura_function:home_handler ---


def home_handler():
    """Handles requests to home_route."""
    return "Home Page - <a href='/login'>Login</a>"


# --- Main Block ---
if __name__ == '__main__':
    app = create_app()

    @app.route('/')
    def home_route_func():
        return home_handler()

    @app.route('/login', methods=['GET', 'POST'])
    def login_route_func():
        if request.method == 'POST':
            return login_handler()
        else:
            return render_login_form()

    @app.route('/logout')
    def logout_route_func():
        return logout_handler()

    @app.route('/protected')
    def protected_route_func():
        return protected_handler()
    app.run(debug=True, port=5000)
