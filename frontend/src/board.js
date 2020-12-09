import React from "react";
import { Tiger, Goat, Piece } from "./pieces";
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
        this.state = {config: config, turn: 'G', goats: 0};
        // this.init = this.init.bind(this)
    }

    init () {
        fetch(
            'http://localhost:5000/new_board', 
            {mode: 'cors', credentials: 'include'}
        ).then(response => 
            response.json()
        ).then(data => {
            this.setState({config: data.config, goats: data.goats});
        });

        window.setInterval(() => this.next_move(), 2000)
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

    componentDidMount() {
        this.init();
    }

    render () {
        const pieces = [];
        this.state.config.forEach((row, i) => {
            row.forEach((val, j) => {
                if (val !== '_') pieces.push([j, i, val])
            });
        });
        return (
            <main id="main">
                <Status goats={this.state.goats} turn={this.state.turn} />
                <div id="game-board">
                    <ul id="pieces-list">
                        {pieces.map(val => {
                            const [x, y, item] = val;
                            if (item === 'G') {
                                return (<Piece x={x} y={y} key={x.toString() + y.toString()}><Goat/></Piece>)
                            } else if (item === 'T') {
                                return (<Piece x={x} y={y} key={x.toString() + y.toString()}><Tiger/></Piece>)
                            }
                        })}
                    </ul>
                </div>
            </main>
        )
    }
}

export default Board;