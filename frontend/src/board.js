import React from "react";
import { Tiger, Goat, Piece, Empty } from "./pieces";
import Status from "./status"
import "./style.scss"

class Board extends React.Component {
    constructor (props) {
        super(props);
        const config = [];
        for (let i = 0; i < 5; i++) {
            const row = [];
            for (let j = 0; j < 5; j++) {
                const random = Math.floor(Math.random() * 3);
                row.push(['G', 'T', '_'][random]);
            }
            config.push(row)
        }
        this.state = {config: config, turn: 'G', goats: 0, selected: null};
        // this.init = this.init.bind(this)
        this.next_move = this.next_move.bind(this)
        this.get_best_moves = this.get_best_moves.bind(this)
    }

    init () {
        fetch(
            'http://localhost:5000/new_board', 
            {mode: 'cors', credentials: 'include'}
            // {mode: 'no-cors', credentials: 'include'},
        ).then(response => 
            response.json()
        ).then(data => {
            this.setState({config: data.config, goats: data.goats});
        });

        // this.loop = window.setInterval(() => this.next_move(), 2000)
    }

    next_move () {
        fetch(
            'http://localhost:5000/next_move?turn='+this.state.turn, 
            {mode: 'cors', credentials: 'include'}
        ).then(response => 
            response.json()
        ).then(data => {
            this.setState({config: data.config, goats: data.goats, turn: this.state.turn === 'G' ? 'T' : 'G'});
        });
    }

    get_best_moves () {
        fetch(
            'http://localhost:5000/get_best_moves?turn='+this.state.turn, 
            {mode: 'cors', credentials: 'include'}
        ).then(response => 
            response.json()
        ).then(data => {
            this.setState({best_moves: data});
            console.log(data)
            // this.setState({config: data.config, goats: data.goats, turn: this.state.turn === 'G' ? 'T' : 'G'});
        });
    }

    make_move (x, y) {
        if (this.state.selected === null) {
            this.setState({selected: [x, y]});
        } else {
            const x_ = this.state.selected[0];
            const y_ = this.state.selected[1];
            const dx = x - x_;
            const dy = y - y_;
            const params = new URLSearchParams({
                turn: this.state.turn,
                x: x_,
                y: y_,
                dx: dx,
                dy: dy,
            });
            fetch(
                'http://localhost:5000/make_move?' + params,
                {mode: 'cors', credentials: 'include'}
            ).then(response =>
                response.json()
            ).then(data => {
                this.setState({config: data.config, goats: data.goats, turn: this.state.turn === 'G' ? 'T' : 'G'});
            });
            this.setState({selected: null});
        }
    }

    componentDidMount() {
        this.init();
    }

    componentWillUnmount() {
        clearInterval(this.loop);
    }

    render () {
        const pieces = [];
        this.state.config.forEach((row, i) => {
            row.forEach((val, j) => {
                pieces.push([j, i, val])
            });
        });
        return (
            <main id="main">
                <Status goats={this.state.goats} turn={this.state.turn} />
                <button onClick={this.next_move}>Next Move</button>
                <button onClick={this.get_best_moves}>Get best moves</button>
                <div id="game-board">
                    <ul id="pieces-list">
                        {pieces.map(val => {
                            const [x, y, item] = val;
                            if (item === 'G') {
                                return (<Piece x={x} y={y} key={x.toString() + y.toString()} onClick={ this.make_move.bind(this, x, y) }><Goat/></Piece>)
                            } else if (item === 'T') {
                                return (<Piece x={x} y={y} key={x.toString() + y.toString()} onClick={ this.make_move.bind(this, x, y) }><Tiger/></Piece>)
                            } else {
                                return (<Piece x={x} y={y} key={x.toString() + y.toString()} onClick={ this.make_move.bind(this, x, y) }><Empty/></Piece>)
                            }
                        })}
                    </ul>
                </div>
            </main>
        )
    }
}

export default Board;