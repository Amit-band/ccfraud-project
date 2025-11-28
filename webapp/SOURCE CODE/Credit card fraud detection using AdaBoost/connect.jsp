<%@ page import="java.sql.*"%>
<%@ page import="java.util.*" %>

<%
    // Read environment variables (used in Docker / Render / Cloud)
    String dbHost = System.getenv("DB_HOST");
    String dbPort = System.getenv("DB_PORT");
    String dbUser = System.getenv("DB_USER");
    String dbPass = System.getenv("DB_PASS");
    String dbName = System.getenv("DB_NAME");

    // Fallbacks (used during local tests)
    if (dbHost == null || dbHost.trim().length() == 0) dbHost = "localhost";
    if (dbPort == null || dbPort.trim().length() == 0) dbPort = "3306";
    if (dbUser == null || dbUser.trim().length() == 0) dbUser = "root";
    if (dbPass == null || dbPass.trim().length() == 0) dbPass = "root";
    if (dbName == null || dbName.trim().length() == 0) dbName = "cc_fraud";

    Connection connection = null;

    try {
        // NEW MYSQL DRIVER
        Class.forName("com.mysql.cj.jdbc.Driver");

        // Build final JDBC URL
        String url = "jdbc:mysql://" + dbHost + ":" + dbPort + "/" + dbName 
                     + "?useSSL=false&allowPublicKeyRetrieval=true&autoReconnect=true";

        // Try connecting
        connection = DriverManager.getConnection(url, dbUser, dbPass);

    } catch(Exception e) {
        // Print DB errors on page (very useful for debugging)
        out.println("<p style='color:red; font-size:18px;'>DB Connection Error: " + e.getMessage() + "</p>");
    }
%>
