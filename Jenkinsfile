// CI pipeline: tests + frontend build (+ optional Selenium). No Docker needed.
def isWin() { return !isUnix() }
def run(String cmd) { if (isUnix()) { sh cmd } else { bat cmd } }
def venvPy() { return isUnix() ? "${env.WORKSPACE}/.venv/bin/python" : "${env.WORKSPACE}\\.venv\\Scripts\\python" }

pipeline {
  agent any
  options { timestamps() }
  parameters {
    booleanParam(name: 'RUN_E2E', defaultValue: false, description: 'Run the local Selenium smoke test (needs Chrome on this machine)')
  }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Backend tests') {
      steps {
        run('python -m venv .venv')
        run("\"${venvPy()}\" -m pip install -r backend/requirements.txt")
        dir('backend') {
          run("\"${venvPy()}\" -m pytest -q --junitxml=test-results.xml")
        }
      }
    }
    stage('Frontend build') {
      steps {
        dir('frontend') {
          run('npm ci')
          run('npm run build')
        }
        archiveArtifacts artifacts: 'frontend/dist/**', fingerprint: true
      }
    }
    stage('E2E (Selenium, optional)') {
      when { expression { params.RUN_E2E } }
      steps {
        run("\"${venvPy()}\" -m pip install -r e2e/requirements.txt")
        run("\"${venvPy()}\" e2e/run_local_e2e.py")
      }
    }
  }
  post {
    always {
      junit allowEmptyResults: true, testResults: 'backend/test-results.xml'
    }
  }
}
