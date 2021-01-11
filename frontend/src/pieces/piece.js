import React from 'react';

class Piece extends React.Component{
    render() {
        return (
            <div className={`single-piece loc-x-${this.props.x} loc-y-${this.props.y}`} onClick={this.props.onClick}>
                {this.props.children}
            </div>
        )
    }
}

export default Piece;