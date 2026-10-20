// CI pipeline: tests + frontend build (+ optional Selenium). No Docker needed.
def isWin() { return !isUnix() }
def runCmd(String cmd) { if (isUnix()) { sh cmd } else { bat cmd } }
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
        runCmd('python -m venv .venv')
        runCmd("\"${venvPy()}\" -m pip install -r backend/requirements.txt")
        dir('backend') {
          runCmd("\"${venvPy()}\" -m pytest -q --junitxml=test-results.xml")
        }
      }
    }
    stage('Frontend build') {
      steps {
        dir('frontend') {
          runCmd('npm ci')
          runCmd('npm run build')
        }
        archiveArtifacts artifacts: 'frontend/dist/**', fingerprint: true
      }
    }
    stage('E2E (Selenium, optional)') {
      when { expression { params.RUN_E2E } }
      steps {
        runCmd("\"${venvPy()}\" -m pip install -r e2e/requirements.txt")
        runCmd("\"${venvPy()}\" e2e/run_local_e2e.py")
      }
    }
  }
  post {
    always {
      junit allowEmptyResults: true, testResults: 'backend/test-results.xml'
    }
  }
}
