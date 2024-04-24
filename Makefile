current: test

run:
	cargo run

test:
	cargo test --release

test1:
	cargo test matrix::solve_qr_test
