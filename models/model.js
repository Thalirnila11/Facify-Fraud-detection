const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const data = new Schema({
   
    email:{
        type: String,
    },
    
},{timestamps: true});

const ldata = mongoose.model("ldata",data);
module.exports = ldata;