<%@ page import="java.sql.*"%>
<%@ page import="java.util.*" %>
<%
    String dbHost = System.getenv("DB_HOST");
    if (dbHost == null || dbHost.trim().length()==0) dbHost = "localhost";
    String dbPort = System.getenv("DB_PORT");
    if (dbPort == null || dbPort.trim().length()==0) dbPort = "3306";
    String dbUser = System.getenv("DB_USER");
    if (dbUser == null || dbUser.trim().length()==0) dbUser = "root";
    String dbPass = System.getenv("DB_PASS");
    if (dbPass == null) dbPass = "root";
    String dbName = System.getenv("DB_NAME");
    if (dbName == null || dbName.trim().length()==0) dbName = "cc_fraud";
    Connection connection = null;
    try {
        Class.forName("com.mysql.cj.jdbc.Driver");
        String url = "jdbc:mysql://" + dbHost + ":" + dbPort + "/" + dbName + "?useSSL=false&allowPublicKeyRetrieval=true";
        connection = DriverManager.getConnection(url, dbUser, dbPass);
    } catch(Exception e) {
        out.println("DB Connection error: " + e.getMessage());
    }
%>
