# 📱 Mobile Access Guide - German Credit Risk Prediction

## ✅ **Network Access Configured Successfully**

Your system is now configured for mobile access on your local network!

---

## 🌐 **Access URLs for Mobile Devices**

### **From Your Phone/Tablet (Same WiFi Network)**

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 **Web App** | `http://192.168.178.25:3000` | Full web interface |
| 🔗 **API** | `http://192.168.178.25:8000` | Direct API access |
| 📚 **API Docs** | `http://192.168.178.25:8000/docs` | Interactive documentation |

### **From Your Computer (Local)**

| Service | URL | Description |
|---------|-----|-------------|
| 🌐 **Web App** | `http://localhost:3000` | Local web interface |
| 🔗 **API** | `http://localhost:8000` | Local API access |

---

## 📱 **Mobile Testing Steps**

### **Step 1: Connect to Same WiFi**
- Ensure your phone is connected to the same WiFi network as your computer
- WiFi network should be the same one your computer uses

### **Step 2: Open Web Browser on Phone**
- Open any web browser (Chrome, Safari, Firefox, etc.)
- Navigate to: `http://192.168.178.25:3000`

### **Step 3: Test the Application**
- Fill out the credit risk assessment form
- Submit and verify you get prediction results
- The app should work exactly like on desktop

---

## 🔧 **Current Configuration**

### **API Container**
```bash
# Running with network binding
docker run -d -p 0.0.0.0:8000:8000 --name credit-risk-api-network ghcr.io/ardas2012/german-credit-risk-prediction:cors-fixed
```

### **Web App**
```bash
# Running with network binding
HOST=0.0.0.0 npm start
```

### **Network Settings**
- **Computer IP**: `192.168.178.25`
- **API Port**: `8000` (accessible from network)
- **Web App Port**: `3000` (accessible from network)
- **CORS**: Enabled for cross-origin requests

---

## 🧪 **Testing from Mobile**

### **Test API Health (Optional)**
Open browser on phone and go to:
```
http://192.168.178.25:8000/health
```

Should return:
```json
{"status":"healthy","model_loaded":true,"preprocessor_loaded":true}
```

### **Test Web App**
1. Open: `http://192.168.178.25:3000`
2. Fill out the form with sample data
3. Submit and verify results appear

---

## 🛠️ **Troubleshooting**

### **Can't Access from Phone**

#### Check 1: Same Network
```bash
# On your computer, check IP
hostname -I
```
Make sure your phone is on the same WiFi network.

#### Check 2: Firewall
```bash
# Check if ports are accessible
sudo ufw status
```
If firewall is active, you may need to allow ports:
```bash
sudo ufw allow 8000
sudo ufw allow 3000
```

#### Check 3: Services Running
```bash
# Check API
curl http://192.168.178.25:8000/health

# Check web app
curl http://192.168.178.25:3000
```

### **IP Address Changed**
If your computer's IP changes, update the web app:

1. **Get new IP**:
   ```bash
   hostname -I
   ```

2. **Update web app** in `credit-risk-webapp/src/components/CreditRiskForm.js`:
   ```javascript
   const response = await axios.post('http://NEW_IP:8000/predict', apiData, {
   ```

3. **Rebuild and restart**:
   ```bash
   cd credit-risk-webapp
   npm run build
   HOST=0.0.0.0 npm start
   ```

---

## 🔒 **Security Notes**

### **Local Network Only**
- This setup only works on your local WiFi network
- External internet users cannot access your system
- This is safe for development and testing

### **Production Deployment**
For internet access, consider:
- Cloud deployment (AWS, Google Cloud, Azure)
- Proper domain and SSL certificates
- Authentication and security measures

---

## 📋 **Current Status**

- ✅ **API**: Network accessible on `192.168.178.25:8000`
- ✅ **Web App**: Network accessible on `192.168.178.25:3000`
- ✅ **CORS**: Configured for cross-origin requests
- ✅ **Mobile Ready**: Can be accessed from phones/tablets

---

## 🎉 **Ready for Mobile Use!**

Your German Credit Risk Prediction system is now accessible from any device on your local network. Simply open `http://192.168.178.25:3000` on your phone's browser to start using it! 