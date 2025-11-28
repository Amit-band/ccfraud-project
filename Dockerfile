FROM tomcat:9.0-jdk11
LABEL maintainer="Amit Band & ChatGPT"
COPY webapp/ /usr/local/tomcat/webapps/ROOT/
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/ || exit 1
