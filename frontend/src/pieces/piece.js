import React from 'react';

class Piece extends React.Component{
    render() {
        const positioning = {
            position: "absolute",
            left: this.props.x * 45 - 5,
            top: this.props.y * 45 - 5,
            fontWeight: 900,
        }
        return (
            <div className="single-piece" style={positioning}>
                {this.props.children}
            </div>
        )
    }
}

export default Piece;