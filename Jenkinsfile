pipeline {
  agent any
  options { timestamps() }
  stages {
    stage('Checkout') {
      steps { checkout scm }
    }
    stage('Backend tests') {
      steps {
        script {
          if (isUnix()) {
            dir('backend') { sh 'python -m pip install -r requirements.txt && python -m pytest -q' }
          } else {
            dir('backend') { bat 'python -m pip install -r requirements.txt && python -m pytest -q' }
          }
        }
      }
    }
    stage('Frontend build') {
      steps {
        script {
          if (isUnix()) {
            dir('frontend') { sh 'npm ci && npm run build' }
          } else {
            dir('frontend') { bat 'npm ci && npm run build' }
          }
        }
      }
    }
    stage('Build and start containers') {
      steps {
        script {
          if (isUnix()) { sh 'docker compose up -d --build' }
          else { bat 'docker compose up -d --build' }
        }
      }
    }
    stage('Selenium smoke test') {
      steps {
        script {
          if (isUnix()) { sh 'docker compose --profile test run --rm e2e' }
          else { bat 'docker compose --profile test run --rm e2e' }
        }
      }
    }
  }
  post {
    always {
      script {
        if (isUnix()) { sh 'docker compose --profile test down --volumes --remove-orphans' }
        else { bat 'docker compose --profile test down --volumes --remove-orphans' }
      }
    }
  }
}
