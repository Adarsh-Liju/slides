---
class: invert
---

# Flatpak

Flatpak is a modern packaging system designed to provide **sandboxed applications** that run consistently across different Linux distributions.

---

## **Flatpak Overview**

Flatpak aims to solve the problem of **distribution fragmentation** by providing a universal application format that works across various Linux distributions while ensuring security through **sandboxing**.

---

# **Core Concepts**

1. **Applications & Runtimes**

   - Applications: The actual Flatpak apps (e.g., Firefox, LibreOffice).
   - Runtimes: Common dependencies shared among applications (e.g., GNOME or KDE runtimes).
   - Reduces redundancy by allowing multiple apps to share runtimes.

2. **Sandboxing**

   - Flatpak applications run in an isolated environment using **Bubblewrap**.
   - Limited access to system files, enhancing security.
   - Permissions (e.g., filesystem, network, USB) must be explicitly granted.

---

# **Core Concepts**

3. **Portals**
   - A mechanism for controlled access to the system.
   - Example: A Flatpak app cannot access files unless granted via the file portal.

---

### **Traditional Package Manager (APT, DNF, Pacman)**

Traditional package managers integrate deeply with the system, making software updates dependent on the OS.

> Open Exalidraw

- Applications rely on **system libraries**.
- A package update may **break dependencies**.
- Requires **root permissions** for installation.

---

### **Flatpak Architecture**

> Open Exalidraw

- Apps are **isolated** (sandboxed).
- **No dependency conflicts**.
- Apps are installed **per-user** without root access.

---

## **Use Cases for Flatpak**

1. **Cross-Distribution Compatibility**

   - Develop once, run everywhere (Ubuntu, Fedora, Arch, etc.).

2. **Security & Sandboxing**

   - Apps run in **restricted environments**, reducing security risks.

3. **Easy Updates & Rollbacks**

   - Automatic updates without affecting system integrity.

4. **Application Bundling**
   - Includes all dependencies, ensuring compatibility.

---
