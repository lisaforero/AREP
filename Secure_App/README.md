# Secure application

## Architecture

The system follows a Decoupled Two-Tier Architecture, ensuring a clear separation between the static content delivery and the secure RESTful API services.

### 1. Web tier (Frontend - Apache HTTP Server)
Deployed on an AWS EC2 instance, this tier is responsible for the user interface.

- Static Assets: Hosts index.html and app.js in /var/www/html.

- Secure Delivery: Uses TLS with certificates managed by Let's Encrypt and Certbot.

- Redirection: Configured via Apache Virtual Hosts to force all HTTP traffic to HTTPS, ensuring no data is transmitted in plain text.

### 2. Application tier (Backend - Spring)
A separate service (or instance) running the Spring framework, which handles the logic and security constraints.

- SecureWeb and SecurityConfig: These classes define the security filter chain. They enforce that every request to the API is encrypted and authenticated.

- AuthController: Manages the identity validation. It processes credentials and ensures that only authorized users can access protected resources.

- HelloController: Serves as the public resource.

- SecureUrlEncoder: This component acts as a Security Testing Client. Its purpose is to programmatically verify the SSL handshake and the Truststore configuration:

  - readURL("https://localhost:5000/hello"): Success Expected. Validates that the application trusts its own certificate (or local CA) stored in myTrustStore.

  - readURL("https://www.google.com"): Failure Expected. Demonstrates a strict security policy where the application only communicates with verified/pinned certificates, rejecting external untrusted authorities.

### 3. Security and trust model
The communication is secured using:

Public Trust: Apache uses Let's Encrypt certificates (arep3.duckdns.org) so that any browser can verify the server's identity.

Internal Trust: The Spring application uses a custom Keystore (ecikeystore.p12) and Truststore (myTrustStore) generated via keytool. This ensures that the backend service itself is a secure identity provider.

### 4. Communication flow
The user accesses https://arep3.duckdns.org:5000/.

Apache serves the index.html and the asynchronous app.js.

app.js performs an asynchronous fetch request to the Spring API.

Spring interceptors (via SecurityConfig) check the authentication headers.

If valid, the /api/secure-data endpoint requests the credentials and then displays the message.

## Security application design & AWS deployment

It was done in the following way:

### 1. Apache instance setup (AWS)
Following the official [Amazon Linux 2023 LAMP guide](https://docs.aws.amazon.com/linux/al2023/ug/ec2-lamp-amazon-linux-2023.html):

1.  **System update and installation:**
    ```bash
    sudo dnf upgrade -y
    sudo dnf install -y httpd wget
    ```
2.  **Service management:**
    ```bash
    sudo systemctl start httpd
    sudo systemctl enable httpd
    sudo systemctl is-enabled httpd
    ```
3.  **Security group:** Port **80 (HTTP)** was opened to allow initial inbound traffic.

### 2. SSL/TLS configuration (Self-Signed)
Following the [SSL/TLS on Amazon Linux 2023 guide](https://docs.aws.amazon.com/linux/al2023/ug/SSL-on-amazon-linux-2023.html):

1.  **Module installation:**
    ```bash
    sudo dnf install openssl mod_ssl
    ```
2.  **Certificate generation:**
    ```bash
    sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/pki/tls/private/apache-selfsigned.key \
    -out /etc/pki/tls/certs/apache-selfsigned.crt
    ```
3.  **Configuration update:** Modified `/etc/httpd/conf.d/ssl.conf`:
    * `SSLCertificateFile /etc/pki/tls/certs/apache-selfsigned.crt`
    * `SSLCertificateKeyFile /etc/pki/tls/private/apache-selfsigned.key`
4.  **Security group:** Port **443 (HTTPS)** was opened.
5.  **When necessary to restore:** `sudo systemctl restart httpd`.

## 3. Domain and Let's Encrypt (Production SSL)
1.  **Dynamic DNS:** Created a domain at **duckdns.org** (e.g., `arep3.duckdns.org`).
2.  **Certbot setup:** Followed [Certbot instructions](https://certbot.eff.org/) for Apache on Linux (pip).
3.  **Virtual host configuration:** Created the following configuration to handle redirection and domain validation:
    ```apache
    <VirtualHost *:80>
        ServerName arep3.duckdns.org
        DocumentRoot /var/www/html
        ErrorLog /var/www/error.log
        CustomLog /var/www/requests.log combined

        RewriteEngine On
        RewriteCond %{HTTPS} off
        RewriteRule ^ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
    </VirtualHost>
    ```
4.  Completed **Step 7** of the Certbot guide to finalize certificate issuance.

### 4. Key management for Spring
A dedicated `keystores` folder was created to manage Java security:

1.  **Generate keystore:**
    ```bash
    keytool -genkeypair -alias ecikeypair -keyalg RSA -keysize 2048 \
    -storetype PKCS12 -keystore ecikeystore.p12 -validity 3650
    ```
2.  **Configuration:** The `resources/application.properties` was updated with the keystore path and credentials.
3.  **Export certificate:**
    ```bash
    keytool -export -keystore ./ecikeystore.p12 -alias ecikeypair -file ecicert.cer
    ```
4.  **Import to truststore:**
    ```bash
    keytool -import -file ./ecicert.cer -alias firstCA -keystore myTrustStore
    ```

### 5. Application deployment & final configuration
1.  **In Apache:** `index.html` and `app.js` were uploaded to `/var/www/html`.
2.  **In Apache:** Modified `/etc/httpd/conf.d/arep3.duckdns.org-le-ssl.conf` for domain-specific SSL fine-tuning.
3.  **In Spring:** * Built locally using `mvn clean package`.
    * Uploaded the `.jar` to the AWS Spring instance.
    * Run command: `java -jar secure-1.0-SNAPSHOT.jar`.

### 6. Integration testing
Final verification was performed using a verbose `curl` command to test authentication and encryption:

```bash
curl -vk -u admin:1306 https://arep3.duckdns.org/api/secure-data
```
