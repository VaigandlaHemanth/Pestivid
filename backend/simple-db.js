// Simple JSON File Database
const fs = require('fs');
const path = require('path');
const bcrypt = require('bcrypt');

const DB_DIR = path.join(__dirname, 'db');

// Ensure db directory exists
if (!fs.existsSync(DB_DIR)) {
    fs.mkdirSync(DB_DIR, { recursive: true });
}

// Helper to read JSON file
function readDB(filename) {
    const filepath = path.join(DB_DIR, filename);
    if (!fs.existsSync(filepath)) {
        fs.writeFileSync(filepath, '[]');
        return [];
    }
    return JSON.parse(fs.readFileSync(filepath, 'utf8'));
}

// Helper to write JSON file
function writeDB(filename, data) {
    const filepath = path.join(DB_DIR, filename);
    fs.writeFileSync(filepath, JSON.stringify(data, null, 2));
}

// User operations
const User = {
    async create(userData) {
        const users = readDB('users.json');
        const hashedPassword = await bcrypt.hash(userData.password, 10);
        const newUser = {
            id: Date.now().toString(),
            name: userData.name,
            email: userData.email.toLowerCase(),
            role: userData.role,
            password: hashedPassword,
            phone: userData.phone || null,
            createdAt: new Date().toISOString()
        };
        users.push(newUser);
        writeDB('users.json', users);
        return newUser;
    },

    findByEmail(email) {
        const users = readDB('users.json');
        return users.find(u => u.email === email.toLowerCase());
    },

    findById(id) {
        const users = readDB('users.json');
        return users.find(u => u.id === id);
    },

    async comparePassword(password, hashedPassword) {
        return await bcrypt.compare(password, hashedPassword);
    },

    getPublicProfile(user) {
        if (!user) return null;
        const { password, ...publicUser } = user;
        return publicUser;
    }
};

// Video operations
const Video = {
    create(videoData) {
        const videos = readDB('videos.json');
        const newVideo = {
            id: Date.now().toString(),
            ...videoData,
            createdAt: new Date().toISOString()
        };
        videos.push(newVideo);
        writeDB('videos.json', videos);
        return newVideo;
    },

    findAll() {
        return readDB('videos.json');
    },

    findById(id) {
        const videos = readDB('videos.json');
        return videos.find(v => v.id === id);
    }
};

// Listing operations
const Listing = {
    create(listingData) {
        const listings = readDB('listings.json');
        const newListing = {
            id: Date.now().toString(),
            ...listingData,
            status: 'active',
            createdAt: new Date().toISOString()
        };
        listings.push(newListing);
        writeDB('listings.json', listings);
        return newListing;
    },

    findAll() {
        return readDB('listings.json');
    },

    findById(id) {
        const listings = readDB('listings.json');
        return listings.find(l => l.id === id);
    }
};

module.exports = { User, Video, Listing };
