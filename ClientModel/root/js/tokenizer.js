class Tokenizer{
    constructor(vacab){
        this.start = vacab['start'];
        this.end = vacab['end'];
        this.unk = vacab['unk'];
        this.mask = vacab['mask'];
        this.pad = vacab['pad'];
        this.char_map = new Map();
        this.index_map = new Map();
        for(let ch in vacab.dict){
            this.char_map.set(ch, vacab.dict[ch]);
            this.index_map.set(vacab.dict[ch], ch);
        }
    }

    token_to_id(token){
        if(this.char_map.has(token)){
            return this.char_map.get(token);
        }
        return this.char_map.get(this.unk);
    }

    id_to_token(id){
        if(this.index_map.has(id)){
            return this.index_map.get(id);
        }
        return this.unk;
    }

    decode(text){
        let ids = [];
        for(let i = 0; i < text.length; i++){
            ids.push(this.token_to_id(text[i]));
        }
        return ids;
    }

    encode(ids){
        let text = ids.map(id => this.id_to_token(id)).join("");
        return text;
    }
}

class InplaceDict{
    constructor(){
        this.padding_len = 3;
        this.password_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ";
        this.ops = [];
        this.ops.push(['k', null]);
        this.ops.push(['d', null]);
        for(let i = 0; i < this.password_letters.length; i++){
            this.ops.push(['s', this.password_letters[i]]);
        }
        for(let i = 0 ; i < this.password_letters.length; i++){
            for(let j = 0; j < this.password_letters.length; j++){
                this.ops.push(['x', this.password_letters[i]+this.password_letters[j]]);
            }
        }
    }

    recover(pwd, edits){
        let ans = pwd.split("");
        for(let i = 0 ; i < this.padding_len; i++){
            ans.push("");
        }
        for(let i = 0; i < edits.length; i++){
            const op = this.ops[edits[i]];
            if(op[0] == 'd'){
                ans[i] = "";
            }
            if(op[0] == 's' || op[0] == 'x'){
                ans[i] = op[1];
            }
        }
        return ans.join("");
    }
}

function PriorityQueue() {
    this.items = [];
    this.uniq = new Set();

    function QueueElement(element, priority) {
        this.element = element;
        this.priority = priority;
    }
    
    this.enqueue = function(element, priority) {
        let name = element.toString();
        if(this.uniq.has(name)){
            return;
        }
        else{
            this.uniq.add(name);
        }
        var queueElement = new QueueElement(element, priority);
        
        if(this.isEmpty()) {
            this.items.push(queueElement);  // {2}
        } else {
            var added = false;
            for(var i = 0; i < this.items.length; i++) {
                if(queueElement.priority >= this.items[i].priority) {
                    this.items.splice(i, 0, queueElement);    // {3}
                    added = true;
                    break;
                }
            }
            if(!added) {    // {4}
                this.items.push(queueElement);
            }
        }
    }
    
    this.dequeue = function() {
        return this.items.shift();
    }
    
    this.front = function() {
        return this.items[0];
    }
    
    this.isEmpty = function() {
        return this.items.length === 0;
    }
    
    this.clear = function() {
        this.items = [];
    }
    
    this.size = function() {
        return this.items.length;
    }
    
    this.print = function() {
        console.log(items.toString());
    }
}

class Evaluator{
    constructor(model, tokenizer){
        this.model = model;
        this.tokenizer = tokenizer;
    }

    encode_text(text){
        let ss = text.split("");
        ss.unshift(this.tokenizer.start);
        ss.push(this.tokenizer.end);
        return this.tokenizer.decode(ss)
    }

    segment_array(ids){
        let res = [];
        for(let i = 0; i < ids.length; i++){
            res.push(0);
        }
        return res;
    }

    encode_input(text){
        let ids = this.encode_text(text);
        let segments = this.segment_array(ids);
        console.log(ids, segments);
        return {
            'Input-Token:0': tf.tensor2d([ids]), 
            'Input-Segment:0': tf.tensor2d([segments])
        };
    }
}

class TPGEvaluator extends Evaluator{
    constructor(model, tokenizer){
        super(model, tokenizer);
        this.padding_len = 3;
    }

    encode_text(text){
        let ss = text.split("");
        ss.unshift(this.tokenizer.start);
        for(let i = 0; i < this.padding_len; i++){
            ss.push(this.tokenizer.start);
        }
        ss.push(this.tokenizer.end);
        return this.tokenizer.decode(ss);
    }

    sorted_prob_list(list, max_index = 10000){
        let ans = [];
        for(let i = 0; i < list.length && i < max_index; i++){
            if(list[i] > 1.0/1000000){
                ans.push({index:i, prob:list[i]});
            }
        }
        return ans.sort((a, b) => {return -a.prob+b.prob});
    }

    mul_prob(top_list, index){
        let prob = 1.0;
        for(let i = 0; i < top_list.length; i++){
            prob *= top_list[i][index[i]].prob;
        }
        return prob;
    }

    top_k(data, k=10){
        data = data[0];
        let top_list = [];
        for(let i = 0; i < data.length; i++){
            top_list.push(this.sorted_prob_list(data[i]));
        }
        let queue = new PriorityQueue();
        let top = [];
        for(let i = 0; i < data.length; i++){
            top.push(0);
        }
        queue.enqueue(top, this.mul_prob(top_list, top));
        
        let path_list = [];
        for(let i = 0; i < k; i++){
            if(queue.size() == 0){
                break;
            }
            let item = queue.dequeue();
            
            path_list.push([item.element.map((a, index) => {return top_list[index][a].index}), item.priority]);
            for(let j = 0; j < data.length; j++){
                let next = item.element.slice();
                next[j] += 1;
                queue.enqueue(next, this.mul_prob(top_list, next));
            }
            
        }
        return path_list;
    }
}


class CPGEvaluator extends Evaluator{
    constructor(model, tokenizer){
        super(model, tokenizer);
    }

    encode_text(text){
        // The text is a character array.
        // let ss = text.split("");
        let ss = text;
        ss.unshift(this.tokenizer.start);
        ss.push(this.tokenizer.end);
        return this.tokenizer.decode(ss);
    }

    mask_text(text){
        let ans = [];
        for(let i = 0; i < text.length; i++){
            let ss = text.split("");
            ss[i]=this.tokenizer.mask;
            ans.push(ss);
        }
        return ans;
    }

    // Main entry 
    encode_input(pwd){
        let masks = this.mask_text(pwd);
        let ids = [];
        let segments = [];
        for(let i = 0; i < masks.length; i++){
            let id = this.encode_text(masks[i]);
            ids.push(id);
            segments.push(this.segment_array(id));
        }
        // console.log(ids, segments);
        return {
            'Input-Token:0': tf.tensor2d(ids), 
            'Input-Segment:0': tf.tensor2d(segments)
        };
    }

    evaluate(pwd, prediction){
        // console.log(prediction);
        let ans = [];
        for(let i = 0; i < pwd.length; i++){
            let ch = pwd[i];
            let index = this.tokenizer.token_to_id(ch);
            // console.log(i, ch, index);
            ans.push(prediction[i][i+1][index]); 
        }
        return ans;
    }
}