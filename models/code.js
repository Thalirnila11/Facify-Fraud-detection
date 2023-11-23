const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const code = new Schema({
    
    code:{
        type:String,
        required: true
    }
    
},{timestamps: true});

const lcode = mongoose.model("code",code);
module.exports = lcode;