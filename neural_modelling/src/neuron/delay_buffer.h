#include <circular_buffer.h>
#include <debug.h>
typedef circular_buffer delay_buffer;

//! \brief Initialise the delay buffer to hold size future delayed items
//! \param[in] size The size of the buffer to allocate
//! \return The newly created buffer, or NULL of out of memory
delay_buffer delay_buffer_initialise(uint32_t size) {
    return circular_buffer_initialize(size * 2);
}

//! \brief Add an item to the delay buffer
//! \param[in] buffer The buffer to add the item to
//! \param[in] delay_to_go The number of cycles to delay this item
//! \param[in] address The address to be read to get this item
//! \param[in] row_length The row length to be read for this item
//! \return true if the item was added, false if no space
bool delay_buffer_add_item(
        delay_buffer buffer, uint32_t delay_to_go, address_t address,
        uint32_t row_length) {
    check(address << 30 == 0, "Address must be divisible by 4");
    check(row_length & 0xFF == row_length, "Row length must be less than 255");
    if (circular_buffer_add(buffer, delay_to_go)) {
        return circular_buffer_add(
            buffer, (((uint32_t) address) << 6) | row_length);
    }
    return false;
}

//! \brief Decrement each delay until one is found to be zero or end of list
//!        is reached.
//! \param[in] buffer The buffer to scan
//! \param[in/out] position A pointer to the initial position to start from,
//!                         or 0xFFFFFFFF to start at the first position.  If
//!                         true is returned, this will contain the position at
//!                         which the first empty item was found.
//! \param[out] address A pointer to be filled with the next address to read
//! \param[out] row_length A pointer to be filled with the next row length
//! \return true if a zero item was found, or false if not.
bool delay_buffer_decrement_and_find_next_zero_delay(
        delay_buffer buffer, uint32_t *position, address_t *address,
        uint32_t* row_length) {
    uint32_t pos = *position;

    // Start the counter at the number of items currently in the list
    if (pos == 0xFFFFFFFF) {
        pos = circular_buffer_size(buffer);
    }

    // Go through the list exactly once
    uint32_t next_delay;
    uint32_t next_address_and_row_length;
    while (pos > 0) {

        // Get the next item in the list and decrement the delay
        circular_buffer_get_next(buffer, &next_delay);
        circular_buffer_get_next(buffer, &next_address_and_row_length);
        next_delay -= 1;

        if (next_delay == 0) {

            // If the next item now has a delay of 0, return it, marking the
            // next position to start from
            *position = pos - 1;
            *address = (address_t)
                ((next_address_and_row_length & 0xFFFFFF00) >> 6);
            *row_length = (next_address_and_row_length & 0xFF);
            return true;
        } else {

            // If the next item doesn't have 0 delay, put it back in the buffer
            circular_buffer_add(buffer, next_delay);
            circular_buffer_add(buffer, next_address_and_row_length);
        }
        pos -= 1;
    }
    return false;
}
