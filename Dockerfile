FROM tomcat:9.0-jdk11
LABEL maintainer="Amit Band & ChatGPT"
COPY "SOURCE CODE/Credit card fraud detection using AdaBoost/" /usr/local/tomcat/webapps/ROOT/
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/ || exit 1
